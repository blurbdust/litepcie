#
# This file is part of LitePCIe.
#
# Copyright (c) 2026 blurbdust <blurbdust@gmail.com>
# SPDX-License-Identifier: BSD-2-Clause

"""
Altera/Intel Stratix 10 L-/H-Tile PCI Express hard IP (Avalon-ST interface), wrapped for LitePCIe.

NOT VERIFIED ON SILICON. Gen2 x8 has been compiled through synthesis, fitter, assembler and timing
analysis for a Microsoft A-2020 (Stratix 10 GX 2800, 1SG280LN2F43E2VG) on Quartus Prime Pro 26.1.1,
and the Avalon-ST, config-space and reset details follow Intel's own RTL, but no link has ever been
trained with this PHY. Protocol details cite Intel doc 683367, Stratix 10 Avalon-ST Interface for
PCIe Solutions User Guide, and Intel's Stratix 10 PCIe example design under
<quartus>/ip/altera/altera_pcie/altera_pcie_s10_ed/.

The hard IP has no pre-generated RTL to point at: `altera_pcie_s10_hip_ast.v.terp` is expanded from
a template by the component's _hw.tcl, so it has to go through Platform Designer. It is generated
from Tcl at build time and cached on a hash of that Tcl, like the Stratix V transceiver
reconfiguration controller in litepcie.phy.svpciephy.

Qword alignment: Stratix V and Cyclone V pad between header and payload so the first payload dword
lands on a lane whose parity matches address bit 2, which is why `svpciephy` carries
`_TXQwordAligner`/`_RXQwordDealigner`. Stratix 10 does NOT: its Avalon-ST payload is packed
immediately after the header, which is LitePCIe's own layout, so no aligner is needed here. The
evidence is Intel's own bridge, `altpcieav_256_data_qwalign_bridge_{rx,tx}.sv` in
altera_pcie_s10_ed, which sits between the hard IP and the legacy Arria 10 style application in the
Avalon-MM example design and is labelled "256 Bit AVST Bridge to Convert A10 AST Signals to S10 AST
Signals". Its RX half *inserts* the padding dword on the way from the hard IP to the application

    3DW: rx_app_data = {rx_fifo_dataq_reg[223:96], 32'h0000_0000, rx_fifo_dataq_reg[95:0]}
    4DW: rx_app_data = {rx_fifo_dataq_reg[223:128], 32'h0000_0000, rx_fifo_dataq_reg[127:0]}

and its TX half removes it again, commented "To Indicate if we need to Remove QW Align the data to
HIP Side". Insertion on the way out of the hard IP means the hard IP itself emits the packed layout.
The padding decision uses the same rule as Stratix V (data present, then bit 2 of header dword 2 for
3DW or dword 3 for 4DW), so the two families differ only in which side of the boundary it is applied
on. Getting this backwards would enumerate and then corrupt every payload, so it is stated from
Intel's RTL rather than assumed from the Stratix V behaviour.

Two more differences from Stratix V worth knowing when reading this next to `svpciephy`:
  - There is no `tx_st_empty`. The hard IP derives the final beat's length from the TLP length
    field, so a partial last beat needs no side-band and LitePCIe's byte enables are only used to
    size the RX side.
  - `rx_st_empty` counts unused *dwords* (3 bits over the 256-bit datapath), not qwords, so RX byte
    enables are exact rather than rounded up to a qword.

Not implemented: Gen3x16 (512-bit datapath), multi-function/SR-IOV, MSI-X, AER, `tx_st_err`,
TX/RX parity, hard IP reconfiguration, the link inspector, and CvP.
"""

import os
import re
import shutil
import hashlib
import subprocess

from migen import *
from migen.genlib.resetsync import AsyncResetSynchronizer

from litex.gen import *

from litex.soc.interconnect import stream
from litex.soc.interconnect.avalon import Native2AvalonST, AvalonST2Native

from litepcie.common import *

# Quartus / Qsys discovery -------------------------------------------------------------------------

def _find_quartus_tool(name, quartus_bindir=None):
    """Locate a Quartus or Qsys binary.

    The Qsys tools live in <quartus>/sopc_builder/bin, which is deliberately *not* on PATH in a
    normal Quartus setup (only <quartus>/bin is), so searching PATH alone is not enough.
    """
    if quartus_bindir is not None:
        for cand in [
            os.path.join(quartus_bindir, name),
            os.path.join(quartus_bindir, "..", "sopc_builder", "bin", name),
        ]:
            cand = os.path.normpath(cand)
            if os.path.exists(cand):
                return cand
    found = shutil.which(name)
    if found is not None:
        return found

    anchor = shutil.which("quartus_sh") or shutil.which("quartus_syn")
    if anchor is not None:
        qbin = os.path.dirname(os.path.realpath(anchor))
        cand = os.path.normpath(os.path.join(qbin, "..", "sopc_builder", "bin", name))
        if os.path.exists(cand):
            return cand

    raise OSError(
        f"Could not find {name}. Qsys tools ship in <quartus>/sopc_builder/bin, which is usually "
        f"not on PATH. Add that directory to PATH, or pass quartus_bindir= to S10PCIEPHY."
    )

# Hard IP generation -------------------------------------------------------------------------------

# `wrala_hwtcl` is the single parameter that fixes link rate, lane count, datapath width and the
# application clock frequency; the strings are the ALLOWED_RANGES of
# <quartus>/ip/altera/altera_pcie/altera_pcie_s10_hip_ast/altera_pcie_s10_hip_ast_parameters.tcl.
# Only the 256-bit modes are listed: Gen3x16 is 512-bit and is not supported by this PHY.
_LINK_MODES = {
    ("gen1",  1) : ("Gen1x1, Interface - 256 bit, 125 MHz",  125e6),
    ("gen1",  2) : ("Gen1x2, Interface - 256 bit, 125 MHz",  125e6),
    ("gen1",  4) : ("Gen1x4, Interface - 256 bit, 125 MHz",  125e6),
    ("gen1",  8) : ("Gen1x8, Interface - 256 bit, 125 MHz",  125e6),
    ("gen1", 16) : ("Gen1x16, Interface - 256 bit, 125 MHz", 125e6),
    ("gen2",  1) : ("Gen2x1, Interface - 256 bit, 125 MHz",  125e6),
    ("gen2",  2) : ("Gen2x2, Interface - 256 bit, 125 MHz",  125e6),
    ("gen2",  4) : ("Gen2x4, Interface - 256 bit, 125 MHz",  125e6),
    ("gen2",  8) : ("Gen2x8, Interface - 256 bit, 125 MHz",  125e6),
    ("gen2", 16) : ("Gen2x16, Interface - 256 bit, 250 MHz", 250e6),
    ("gen3",  1) : ("Gen3x1, Interface - 256 bit, 125 MHz",  125e6),
    ("gen3",  2) : ("Gen3x2, Interface - 256 bit, 125 MHz",  125e6),
    ("gen3",  4) : ("Gen3x4, Interface - 256 bit, 125 MHz",  125e6),
    ("gen3",  8) : ("Gen3x8, Interface - 256 bit, 250 MHz",  250e6),
}

def _build_hip_qsys_tcl(name, instance, device, family, link_mode, bar0_size, vendor_id, device_id,
    revision_id, class_code, subsys_vendor_id, subsys_device_id, max_payload_size, msi_enable):
    """Tcl that builds the hard IP as a Platform Designer system.

    Every parameter left at its default is left out, so the Tcl reads as the delta from Intel's
    defaults. The values that are set follow Intel's Gen2 x8 Avalon-ST example design
    (altera_pcie_s10_ed/example_design/s10/ep_g2x8_ast.qar) and the hardware-verified A-2020
    design at https://github.com/quake0day/msa2020-stratix10.
    """
    parameters = {
        # Link and application interface.
        "wrala_hwtcl"                     : link_mode,
        "app_interface_width_hwtcl"       : "256-bit",
        "pll_refclk_freq_hwtcl"           : "100 MHz",
        "virtual_rp_ep_mode_hwtcl"        : "Native Endpoint",
        "use_ast_parity_hwtcl"            : 0,
        "use_pll_lock_hwtcl"              : 1,

        # `clr_st` is only driven when this is on; it is the reset Intel's example design applies to
        # the application, see altpcied_s10_hwtcl.sv line 379.
        "enable_avst_reset_hwtcl"         : 1,

        # The lanes on the A-2020 edge connector are in reverse order on the two halves of the x16
        # slot; auto lane flip lets the hard IP work either way round.
        "pf0_auto_lane_flip_ctrl_en_hwtcl": "enable",

        # BAR0 only. Stratix 10 offers no 64-bit non-prefetchable BAR: the choices are "Disabled",
        # "64-bit prefetchable memory" and "32-bit non-prefetchable memory" (parameters.tcl:292).
        "pf0_bar0_type_hwtcl"             : "64-bit prefetchable memory",
        "pf0_bar0_address_width_hwtcl"    : log2_int(bar0_size),

        # Identity.
        "pf0_pci_type0_vendor_id_hwtcl"   : vendor_id,
        "pf0_pci_type0_device_id_hwtcl"   : device_id,
        "pf0_revision_id_hwtcl"           : revision_id,
        "pf0_class_code_hwtcl"            : class_code,
        "pf0_subsys_vendor_id_hwtcl"      : subsys_vendor_id,
        "pf0_subsys_dev_id_hwtcl"         : subsys_device_id,
        "virtual_maxpayload_size_hwtcl"   : max_payload_size,
        "total_pf_count_hwtcl"            : 1,
        "enable_multi_func_hwtcl"         : 0,
        "enable_sriov_hwtcl"              : 0,

        # MSI: one vector, no MSI-X. LitePCIe's MSI block drives app_msi_req/app_msi_ack.
        "virtual_pf0_msi_enable_hwtcl"    : 1 if msi_enable else 0,
        "virtual_pf0_msix_enable_hwtcl"   : 0,
        "pf0_pci_msi_multiple_msg_cap_hwtcl" : 1,

        # No reconfiguration paths and no link inspector: each would add an Avalon-MM port with
        # nothing to drive it.
        "hip_reconfig_hwtcl"              : 0,
        "xcvr_reconfig_hwtcl"             : 0,
        "pcie_link_inspector_hwtcl"       : 0,

        # Example designs. Generating them costs minutes and produces a second top level that
        # Quartus would try to compile alongside the SoC.
        "enable_example_design_qii_hwtcl"   : 0,
        "enable_example_design_sim_hwtcl"   : 0,
        "enable_example_design_synth_hwtcl" : 0,
        "enable_example_design_tb_hwtcl"    : 0,
    }
    tcl = [
        "package require -exact qsys 16.0",
        f"create_system {name}",
        f"set_project_property DEVICE_FAMILY {{{family}}}",
        f"set_project_property DEVICE {{{device}}}",
        f"add_instance {instance} altera_pcie_s10_hip_ast",
    ]
    for k in sorted(parameters):
        tcl.append(f"set_instance_parameter_value {instance} {{{k}}} {{{parameters[k]}}}")
    # AUTO_EXPORT rather than one add_interface per interface: the hard IP has 322 ports across 18
    # interfaces, and exporting by hand only creates opportunities to miss one.
    tcl.append(f"set_instance_property {instance} AUTO_EXPORT true")
    tcl.append(f"save_system {name}.qsys")
    return "\n".join(tcl) + "\n"

def _collect_qips(output_dir):
    """Every .qip the generation wrote, so the whole hierarchy reaches Quartus.

    On 26.1.1 this is a single file that already lists the subcores (transceiver PHY, fPLL, PCIe
    native PHY) and their SDCs, but Platform Designer splits a system into one .qip per instance
    when it is not a standalone IP variation, and the system .qip does not reference those. Walking
    for them costs nothing and a missed one would synthesise as a black box.
    """
    qips = []
    for root, _, files in os.walk(output_dir):
        qips += [os.path.join(root, f) for f in files if f.endswith(".qip")]
    if not qips:
        raise OSError(f"[S10PCIEPHY] no .qip found under {output_dir}.")
    return sorted(qips)

def generate_hip(name, instance, device, family, output_dir, quartus_bindir=None, force=False,
    **params):
    """Generate the PCIe hard IP as a Platform Designer system and return (qip paths, top Verilog).

    Cached on a hash of everything that affects the output, like litepcie.phy.svpciephy. Generation
    takes several minutes, so the cache is what makes an edit-rebuild cycle bearable.
    """
    tcl = _build_hip_qsys_tcl(name, instance, device, family, **params)
    _CACHE_VERSION = "1"
    hash = hashlib.sha256((_CACHE_VERSION + tcl).encode()).hexdigest()[:16]

    output_dir = os.path.abspath(output_dir)
    stampfile  = os.path.join(output_dir, f"{name}.litex_hash")
    top        = os.path.join(output_dir, name, "synth", f"{name}.v")

    if not force and os.path.exists(top) and os.path.exists(stampfile):
        if open(stampfile).read().strip() == hash:
            print(f"[S10PCIEPHY] Reusing cached PCIe hard IP in {output_dir} (hash {hash}).")
            return _collect_qips(output_dir), top

    # Drop what a previous generation left. The .qip set is collected by walking output_dir, so a
    # stale one from different parameters would otherwise be handed to Quartus alongside the new
    # ones. Only the paths Platform Designer creates are removed, never output_dir itself.
    for stale in [name, "ip", "qdb", "dni", f"{name}.ip", f"{name}.qsys", f"{name}.qpf",
                  f"{name}.qsf", stampfile]:
        path = os.path.join(output_dir, stale)
        shutil.rmtree(path, ignore_errors=True)
        if os.path.isfile(path):
            os.remove(path)

    qsys_script   = _find_quartus_tool("qsys-script",   quartus_bindir)
    qsys_generate = _find_quartus_tool("qsys-generate", quartus_bindir)

    os.makedirs(output_dir, exist_ok=True)
    tclfile = os.path.join(output_dir, f"{name}.tcl")
    with open(tclfile, "w") as f:
        f.write(tcl)

    print(f"[S10PCIEPHY] Generating {name} ({params['link_mode']}) for {device}. "
          f"This takes several minutes.")
    def run(cmd):
        r = subprocess.run(cmd, cwd=output_dir)
        if r.returncode != 0:
            raise OSError(
                f"[S10PCIEPHY] {os.path.basename(cmd[0])} failed (exit {r.returncode}).\n"
                f"  cmd: {' '.join(cmd)}\n  cwd: {output_dir}\n"
                f"Rerun it by hand there to see the full Qsys log."
            )
    run([qsys_script, f"--script={name}.tcl"])

    # A system holding a single instance is a "standalone IP variation", which Qsys refuses to save
    # as .qsys and writes as .ip instead ("Output standalone IP variation files of type Qsys are not
    # supported"). Which one appeared is checked rather than assumed, so a future Qsys that honours
    # the .qsys request does not break this.
    system = None
    for ext in [".ip", ".qsys"]:
        cand = os.path.join(output_dir, name + ext)
        if os.path.exists(cand):
            system = name + ext
            break
    if system is None:
        raise OSError(f"[S10PCIEPHY] qsys-script wrote neither {name}.ip nor {name}.qsys in "
                      f"{output_dir}.")

    run([qsys_generate, system, "--synthesis=VERILOG", f"--family={family}", f"--part={device}"])
    if not os.path.exists(top):
        raise OSError(f"[S10PCIEPHY] generation reported success but {top} is missing.")

    with open(stampfile, "w") as f:
        f.write(hash)
    return _collect_qips(output_dir), top

def probe_ports(top_verilog, name):
    """Every port of the generated wrapper, as {port: width}.

    Read rather than hardcoded for two reasons. `tl_cfg_add` is 4 or 5 bits depending on the die
    type (altera_pcie_s10_hip_ast_interfaces.tcl, elaborate_config_tl), and a signal one bit too
    wide would be truncated silently at the instance boundary, decoding the configuration space at
    the wrong addresses. And the full set lets the instantiation be checked for completeness: a hard
    IP input this PHY forgot would be left floating and misbehave in a way no synthesis message
    points at.
    """
    src   = open(top_verilog).read()
    start = src.index(f"module {name} (")
    end   = re.search(r"\n\s*\);", src[start:])
    if end is None:
        raise ValueError(f"Could not find the end of module {name}'s port list in {top_verilog}.")
    header = src[start:start + end.start()]
    ports  = {}
    for line in header.splitlines():
        m = re.match(r"\s*(?:input|output)\s+wire\s+(?:\[\s*(\d+)\s*:\s*0\s*\]\s+)?(\w+)", line)
        if m is not None:
            ports[m.group(2)] = 1 if m.group(1) is None else int(m.group(1)) + 1
    if not ports:
        raise ValueError(f"Could not read any port off {top_verilog}.")
    return ports

# S10PCIEPHY ---------------------------------------------------------------------------------------

class S10PCIEPHY(LiteXModule):
    endianness    = "little"
    qword_aligned = False # Stratix 10 packs payload straight after the header; see the module
                          # docstring for the evidence.

    def __init__(self, platform, pads, ninit_done, data_width=256, cd="sys",
        # Link configuration.
        speed          = "gen2",
        nlanes         = 8,
        # PCIe hard block parameters.
        bar0_size      = 0x100000,
        vendor_id      = 0x1172,   # Altera.
        device_id      = 0xe001,
        revision_id    = 0x00,     # LitePCIe's kernel driver rejects any non-zero PCI revision.
        class_code     = 0xff0000, # Unassigned class.
        subsys_vendor_id = 0x1172,
        subsys_device_id = 0xe001,
        max_payload_size = 256,
        msi_enable       = True,
        # Toolchain.
        device         = None,
        family         = "Stratix 10",
        name           = "s10pcie",
        instance       = "hip",
        output_dir     = None,
        quartus_bindir = None,
    ):
        # Streams ---------------------------------------------------------------------------------
        self.sink   = stream.Endpoint(phy_layout(data_width))
        self.source = stream.Endpoint(phy_layout(data_width))
        self.msi    = stream.Endpoint(msi_layout())

        # Parameters/Locals ------------------------------------------------------------------------
        self.pads             = pads
        self.platform         = platform
        self.data_width       = data_width

        self.id               = Signal(16, reset_less=True)
        self.bar0_size        = bar0_size
        self.bar0_mask        = get_bar_mask(bar0_size)
        self.max_request_size  = Signal(16, reset_less=True)
        self.max_payload_size  = Signal(16, reset_less=True)

        self.external_hard_ip = False

        # Checks -----------------------------------------------------------------------------------
        if (speed, nlanes) not in _LINK_MODES:
            raise ValueError(f"S10PCIEPHY: {speed} x{nlanes} is not a supported hard IP mode "
                             f"(supported: {sorted(_LINK_MODES)}).")
        if data_width != 256:
            raise ValueError(
                "S10PCIEPHY: the L-/H-Tile Avalon-ST hard IP has a 256-bit application interface in "
                "every mode this PHY supports; only Gen3x16 is 512-bit and that is not implemented.")
        if len(pads.tx_p) != nlanes or len(pads.rx_p) != nlanes:
            # Not just "at least nlanes": every pin of a requested resource gets a location
            # assignment, and a transceiver pin that is constrained but left without a driver is a
            # fitter error, not a warning. The caller has to hand over exactly the lanes in use.
            raise ValueError(
                f"S10PCIEPHY: pads carry {len(pads.tx_p)} tx / {len(pads.rx_p)} rx lanes but "
                f"{nlanes} are driven; the rest would be constrained with nothing driving them.")

        link_mode, self.pcie_clk_freq = _LINK_MODES[(speed, nlanes)]

        device = device if device is not None else platform.device
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "build", "s10pcie_ip")

        # # #

        # Hard IP generation -----------------------------------------------------------------------
        qips, top = generate_hip(
            name             = name,
            instance         = instance,
            device           = device,
            family           = family,
            output_dir       = output_dir,
            quartus_bindir   = quartus_bindir,
            link_mode        = link_mode,
            bar0_size        = bar0_size,
            vendor_id        = vendor_id,
            device_id        = device_id,
            revision_id      = revision_id,
            class_code       = class_code,
            subsys_vendor_id = subsys_vendor_id,
            subsys_device_id = subsys_device_id,
            max_payload_size = max_payload_size,
            msi_enable       = msi_enable,
        )
        for qip in qips:
            platform.add_ip(qip)
        widths = self.hip_ports = probe_ports(top, name)

        # Clock domain -----------------------------------------------------------------------------
        # The application clock is coreclkout_hip; unlike Stratix V there is no separate pld_clk to
        # feed back. `clr_st` is the reset Intel's example design applies to application logic when
        # enable_avst_reset is on (altpcied_s10_hwtcl.sv line 379), and it is asynchronous to
        # coreclkout_hip, so it is resynchronised here exactly as their reset_delay_sync does.
        self.cd_pcie = ClockDomain()

        coreclkout_hip = Signal()
        clr_st         = Signal()
        reset_status   = Signal()
        self.comb += self.cd_pcie.clk.eq(coreclkout_hip)
        self.specials += AsyncResetSynchronizer(self.cd_pcie, clr_st)

        # TX CDC (FPGA --> HOST) -------------------------------------------------------------------
        if cd == "pcie":
            tx_st = self.sink
        else:
            tx_buffer = stream.Buffer(phy_layout(data_width))
            tx_buffer = ClockDomainsRenamer(cd)(tx_buffer)
            tx_cdc    = stream.AsyncFIFO(phy_layout(data_width), 32)
            tx_cdc    = ClockDomainsRenamer({"write": cd, "read": "pcie"})(tx_cdc)
            self.submodules += tx_buffer, tx_cdc
            self.comb += [
                self.sink.connect(tx_buffer.sink),
                tx_buffer.source.connect(tx_cdc.sink),
            ]
            tx_st = tx_cdc.source

        # RX CDC (HOST --> FPGA) -------------------------------------------------------------------
        if cd == "pcie":
            rx_st = self.source
        else:
            rx_cdc    = stream.AsyncFIFO(phy_layout(data_width), 32)
            rx_cdc    = ClockDomainsRenamer({"write": "pcie", "read": cd})(rx_cdc)
            rx_buffer = stream.Buffer(phy_layout(data_width))
            rx_buffer = ClockDomainsRenamer(cd)(rx_buffer)
            self.submodules += rx_buffer, rx_cdc
            self.comb += [
                rx_cdc.source.connect(rx_buffer.sink),
                rx_buffer.source.connect(self.source),
            ]
            rx_st = rx_cdc.sink

        # MSI CDC (FPGA --> HOST) ------------------------------------------------------------------
        if cd == "pcie":
            cfg_msi = self.msi
        else:
            msi_cdc = stream.AsyncFIFO(msi_layout(), 4)
            msi_cdc = ClockDomainsRenamer({"write": cd, "read": "pcie"})(msi_cdc)
            self.submodules += msi_cdc
            self.comb += self.msi.connect(msi_cdc.sink)
            cfg_msi = msi_cdc.source

        # Configuration space decode ----------------------------------------------------------------
        # tl_cfg_ctl is a 32-bit window multiplexed over 16 addresses selected by tl_cfg_add, with
        # the field mapping of Intel's own consumers: altpcierd_tl_cfg_sample.v, which the Stratix 10
        # example design instantiates, and altpciexpav_256_cfg_status.v, which the Stratix 10
        # Avalon-MM bridge feeds. Both take only the low four bits of tl_cfg_add (it is five bits
        # wide on this die type), so the window is decoded masked here too.
        #
        # There is no write strobe port on this hard IP, and none is needed: altpcierd_tl_cfg_sample
        # gates sampling on one for Arria V and Cyclone V only (its HIP_SV parameter), and on every
        # later family, Stratix 10 included, that condition is unconditionally true and the window is
        # simply sampled every cycle.
        tl_cfg_add = Signal(widths["tl_cfg_add"])
        tl_cfg_ctl = Signal(32)

        def convert_size(command, size):
            cases = {}
            value = 128
            for i in range(6):
                cases[i] = size.eq(value)
                value = value*2
            return Case(command, cases)

        bus_number      = Signal(8)
        device_number   = Signal(5)
        function_number = Signal(3)
        dcommand        = Signal(16)

        self.bus_number      = bus_number
        self.device_number   = device_number
        self.function_number = function_number

        self.sync.pcie += [
            # tl_cfg_add == 0 : Device Control register in bits [31:16] (cfg_devcsr[15:0]).
            If(tl_cfg_add[:4] == 0x0,
                dcommand.eq(tl_cfg_ctl[16:32]),
            ),
            # tl_cfg_add == 15 : cfg_busdev[12:0] = {bus[7:0], device[4:0]}.
            If(tl_cfg_add[:4] == 0xf,
                device_number.eq(tl_cfg_ctl[0:5]),
                bus_number.eq(tl_cfg_ctl[5:13]),
            ),
        ]

        self.sync.pcie += [
            convert_size(dcommand[5:8],   self.max_payload_size),
            convert_size(dcommand[12:15], self.max_request_size),
            self.id.eq(Cat(function_number, device_number, bus_number)),
        ]
        # Only one function is enabled on the hard IP, so the function number is always 0.
        self.comb += function_number.eq(0)

        # Native <-> Avalon-ST ----------------------------------------------------------------------
        # Both directions carry readyLatency 3 on Stratix 10
        # (altera_pcie_s10_hip_ast_interfaces.tcl lines 898 and 905), unlike Stratix V where TX is 2.
        tx_n2av = ClockDomainsRenamer("pcie")(Native2AvalonST(phy_layout(data_width), latency=3))
        rx_av2n = ClockDomainsRenamer("pcie")(AvalonST2Native(phy_layout(data_width), latency=3))
        self.submodules += tx_n2av, rx_av2n

        self.comb += [
            tx_st.connect(tx_n2av.sink),
            rx_av2n.source.connect(rx_st),
        ]

        tx_avst = tx_n2av.source
        rx_avst = rx_av2n.sink

        # `rx_st_empty` is the number of unused dwords in the final beat of a packet and is only
        # meaningful with rx_st_eop, so byte enables are all-ones elsewhere. TX needs no equivalent:
        # this hard IP has no tx_st_empty and takes the payload length from the TLP header.
        ndwords  = data_width//32
        rx_empty = Signal(widths["rx_st_empty"])
        self.comb += [
            rx_avst.be.eq(2**(data_width//8)-1),
            If(rx_avst.last,
                Case(rx_empty, {d : rx_avst.be.eq(2**(data_width//8 - 4*d)-1)
                                for d in range(ndwords)}),
            ),
        ]

        # Hard IP ------------------------------------------------------------------------------------
        # Tie-offs are emitted at the port's own width. Migen writes a bare 0 as a 1-bit literal, and
        # Quartus then reports "actual bit length 1 differs from formal bit length N" for every wide
        # input that is tied off, which on this hard IP is about a hundred warnings for the PIPE
        # interface alone; a genuine width mismatch would be lost in them.
        def tie(port, value=0):
            return Constant(value, widths[port])

        self.hip_params = dict(
            # Clocks. refclk is the 100 MHz HCSL reference from the edge connector.
            i_refclk         = pads.clk_p,
            o_coreclkout_hip = coreclkout_hip,

            # Reset. npor is the application's power-on reset and pin_perst is PERST# from the
            # connector; both are active low. ninit_done comes from the Reset Release IP and holds
            # the hard IP off until device initialisation has finished, which on Stratix 10 happens
            # after the fabric is already running.
            i_npor           = ~ResetSignal(cd),
            i_pin_perst      = pads.rst_n,
            i_ninit_done     = ninit_done,
            o_reset_status   = reset_status,
            o_clr_st         = clr_st,
            o_pld_clk_inuse  = Signal(),
            o_testin_zero    = Signal(),

            # Test/simulation control. Unlike Stratix V, test_in is all zeroes in hardware; Intel's
            # example design and the A-2020 prior art both tie it off.
            i_test_in                  = tie("test_in"),
            i_simu_mode_pipe           = tie("simu_mode_pipe"),
            i_sim_pipe_pclk_in         = tie("sim_pipe_pclk_in"),
            i_sim_pipe_mask_tx_pll_lock = tie("sim_pipe_mask_tx_pll_lock"),

            # Config space.
            o_tl_cfg_add     = tl_cfg_add,
            o_tl_cfg_ctl     = tl_cfg_ctl,
            o_tl_cfg_func    = Signal(2),
            i_app_err_valid  = tie("app_err_valid"),
            i_app_err_hdr    = tie("app_err_hdr"),
            i_app_err_info   = tie("app_err_info"),

            # Function Level Reset: not handled, so never acknowledged.
            i_flr_pf_done    = tie("flr_pf_done"),
            o_flr_pf_active  = Signal(2),

            # MSI.
            i_app_msi_req    = cfg_msi.valid,
            o_app_msi_ack    = cfg_msi.ready,
            i_app_msi_num    = tie("app_msi_num"),
            i_app_msi_tc     = tie("app_msi_tc"),
            i_app_int_sts    = tie("app_int_sts"),

            # Power management: unused.
            i_apps_pm_xmt_pme     = tie("apps_pm_xmt_pme"),
            i_apps_ready_entr_l23 = tie("apps_ready_entr_l23"),
            i_apps_pm_xmt_turnoff = tie("apps_pm_xmt_turnoff"),
            i_app_init_rst        = tie("app_init_rst"),
            i_app_xfer_pending    = tie("app_xfer_pending"),
            o_pm_linkst_in_l1     = Signal(),
            o_pm_linkst_in_l0s    = Signal(),
            o_pm_state            = Signal(3),
            o_pm_dstate           = Signal(3),

            # RX Avalon-ST.
            o_rx_st_valid    = rx_avst.valid,
            o_rx_st_sop      = rx_avst.first,
            o_rx_st_eop      = rx_avst.last,
            i_rx_st_ready    = rx_avst.ready,
            o_rx_st_data     = rx_avst.dat,
            o_rx_st_empty    = rx_empty,
            o_rx_st_bar_range = Signal(3),

            # TX Avalon-ST. tx_st_err marks a TLP the application wants the hard IP to discard;
            # LitePCIe never emits one.
            i_tx_st_valid    = tx_avst.valid,
            i_tx_st_sop      = tx_avst.first,
            i_tx_st_eop      = tx_avst.last,
            o_tx_st_ready    = tx_avst.ready,
            i_tx_st_data     = tx_avst.dat,
            i_tx_st_err      = tie("tx_st_err"),
        )

        # Status, and the TX credit interface LitePCIe does not use. Wired to signals with no load
        # rather than omitted, so the port check in do_finalize() can stay a strict equality;
        # synthesis removes them. Kept in a dict so a target can watch link_up or ltssmstate.
        self.hip_status = {
            "int_status"            : Signal(widths["int_status"]),
            "int_status_common"     : Signal(3),
            "derr_cor_ext_rpl"      : Signal(),
            "derr_rpl"              : Signal(),
            "derr_cor_ext_rcv"      : Signal(),
            "derr_uncor_ext_rcv"    : Signal(),
            "rx_par_err"            : Signal(),
            "tx_par_err"            : Signal(),
            "ltssmstate"            : Signal(widths["ltssmstate"]),
            "link_up"               : Signal(),
            "lane_act"              : Signal(widths["lane_act"]),
            "currentspeed"          : Signal(2),
            "serdes_pll_locked"     : Signal(),
            "sim_pipe_rate"         : Signal(2),
            "sim_ltssmstate"        : Signal(6),
            "tx_cdts_type"          : Signal(2),
            "tx_data_cdts_consumed" : Signal(),
            "tx_hdr_cdts_consumed"  : Signal(),
            "tx_cdts_data_value"    : Signal(2),
            "tx_cpld_cdts"          : Signal(12),
            "tx_pd_cdts"            : Signal(12),
            "tx_npd_cdts"           : Signal(12),
            "tx_cplh_cdts"          : Signal(8),
            "tx_ph_cdts"            : Signal(8),
            "tx_nph_cdts"           : Signal(8),
        }
        self.hip_params.update({f"o_{k}": v for k, v in self.hip_status.items()})

        # pld_core_ready tells the hard IP the fabric side is up. Intel's example design drives it
        # from serdes_pll_locked (altpcied_s10_hwtcl.sv line 1005).
        self.hip_params.update(i_pld_core_ready = self.hip_status["serdes_pll_locked"])

        # PIPE interface: hardware mode drives the serial lanes, so the PIPE ports exist only for
        # simulation. Inputs are tied off and outputs left open, per Intel's example design.
        for lane in range(8):
            for port in [
                "rxdata", "rxdatak", "phystatus", "rxvalid", "rxstatus", "rxelecidle", "rxsynchd",
                "rxblkst", "rxdataskip", "dirfeedback",
            ]:
                self.hip_params[f"i_{port}{lane}"] = tie(f"{port}{lane}")
            for port, width in [
                ("txdata", 32), ("txdatak", 4), ("txcompl", 1), ("txelecidle", 1), ("txdetectrx", 1),
                ("powerdown", 2), ("txmargin", 3), ("txdeemph", 1), ("txswing", 1), ("txsynchd", 2),
                ("txblkst", 1), ("txdataskip", 1), ("rate", 2), ("rxpolarity", 1),
                ("currentrxpreset", 3), ("currentcoeff", 18), ("rxeqeval", 1),
                ("rxeqinprogress", 1), ("invalidreq", 1),
            ]:
                self.hip_params[f"o_{port}{lane}"] = Signal(width)

        # Serial lanes. Only the positive halves are named: Quartus infers the negative of a
        # transceiver pair from the package.
        for i in range(nlanes):
            self.hip_params[f"i_rx_in{i}"]  = pads.rx_p[i]
            self.hip_params[f"o_tx_out{i}"] = pads.tx_p[i]

        # Constraints ---------------------------------------------------------------------------------
        # coreclkout_hip is unrelated to the SoC clocks; every crossing here is through an async FIFO
        # or a reset synchroniser. The clock has to be named the way Intel's SDC names it, not the
        # way LiteX would: `coreclkout_hip` is `pllnphy_tx_clkout[0]`, transceiver channel 0's TX
        # clkout, and the transceiver SDC creates it as `<hierarchy>|xcvr_hip_native|ch0`. Nothing in
        # the design carries a clock called `pcie_clk`, so the `add_false_path_constraints()` that
        # `SoC.add_pcie()` emits for cd_pcie matches nothing and this is the constraint that bites.
        pcie_core_clk = "*|xcvr_hip_native|*"
        platform.toolchain.additional_sdc_commands += [
            f"set_false_path -from [get_clocks {{{cd}_clk}}] -to [get_clocks {{{pcie_core_clk}}}]",
            f"set_false_path -from [get_clocks {{{pcie_core_clk}}}] -to [get_clocks {{{cd}_clk}}]",
        ]

        # Constrain the 100 MHz PCIe reference clock at the pin, otherwise the transceiver PLL has
        # no base clock and everything derived from it is unconstrained. Intel's example SDC uses
        # `derive_pll_clocks -create_base_clocks`; LiteX emits `-use_net_name`, and derive_pll_clocks
        # must only be applied once per project, so the base clock is declared here instead.
        platform.add_period_constraint(pads.clk_p, 1e9/100e6)

        # Timing analysis of transceiver-derived clocks is meaningless without this, and Quartus
        # only warns if it is missing. Guarded because other Altera IP wrappers add it too and it
        # must not be issued twice.
        if "derive_clock_uncertainty" not in platform.toolchain.additional_sdc_commands:
            platform.toolchain.additional_sdc_commands.append("derive_clock_uncertainty")

        self.name = name

    def do_finalize(self):
        driven  = {k[2:] for k in self.hip_params}
        missing = set(self.hip_ports) - driven
        unknown = driven - set(self.hip_ports)
        if missing or unknown:
            raise ValueError(
                f"S10PCIEPHY: instantiation does not match the generated wrapper.\n"
                f"  not driven here : {sorted(missing)}\n"
                f"  not on the IP   : {sorted(unknown)}\n"
                f"This Quartus generated a different port set than the one this PHY was written "
                f"against; a floating hard IP input does not show up as a synthesis error.")
        self.specials += Instance(self.name, **self.hip_params)
