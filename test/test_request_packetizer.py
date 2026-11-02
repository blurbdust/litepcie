#
# This file is part of LitePCIe.
#
# Copyright (c) 2026 Florent Kermarrec <florent@enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause

import math
import unittest

from migen.sim import passive, run_simulation

from litepcie.tlp.packetizer import LitePCIeTLPPacketizer


class TestRequestPacketizer(unittest.TestCase):
    def packetize_write(self, data_width, length):
        dut = LitePCIeTLPPacketizer(
            data_width    = data_width,
            endianness    = "big",
            address_width = 64,
            capabilities  = ["REQUEST"],
        )
        beats = []

        @passive
        def monitor():
            while True:
                yield dut.source.ready.eq(1)
                if (yield dut.source.valid) and (yield dut.source.ready):
                    beats.append({
                        "be"    : (yield dut.source.be),
                        "first" : (yield dut.source.first),
                        "last"  : (yield dut.source.last),
                    })
                yield

        def stimulus():
            sink = dut.req_sink
            dwords_per_beat = data_width//32
            nbeats = math.ceil(length/dwords_per_beat)

            yield sink.valid.eq(0)
            yield sink.we.eq(1)
            yield sink.adr.eq(0x1_0000_1000)
            yield sink.len.eq(length)
            yield sink.req_id.eq(0x1234)
            yield sink.tag.eq(0x5a)
            yield

            for beat in range(nbeats):
                yield sink.dat.eq(0x12345678 << (32*(beat % dwords_per_beat)))
                yield sink.first.eq(beat == 0)
                yield sink.last.eq(beat == (nbeats - 1))
                yield sink.valid.eq(1)
                yield
                while not (yield sink.ready):
                    yield

            yield sink.valid.eq(0)
            for _ in range(30):
                yield

        run_simulation(dut, [stimulus(), monitor()])
        return beats

    def test_write_keep_matches_length(self):
        for data_width in [128, 256, 512]:
            dwords_per_beat = data_width//32
            lengths = sorted(set([
                1,
                dwords_per_beat//2,
                dwords_per_beat,
                dwords_per_beat + dwords_per_beat//2,
            ]))
            for length in lengths:
                with self.subTest(data_width=data_width, length=length):
                    beats = self.packetize_write(data_width, length)
                    valid_dwords = sum(
                        1
                        for beat in beats
                        for lane in range(data_width//32)
                        if ((beat["be"] >> (4*lane)) & 0xf) == 0xf
                    )

                    self.assertEqual(valid_dwords, 4 + length)
                    self.assertEqual([beat["last"] for beat in beats].count(1), 1)

    def test_512b_32byte_tail_fits_one_rq_beat(self):
        beats = self.packetize_write(data_width=512, length=8)

        self.assertEqual(len(beats), 1)
        self.assertEqual(beats[0]["first"], 1)
        self.assertEqual(beats[0]["last"], 1)
        self.assertEqual(beats[0]["be"], (1 << (16 + 32)) - 1)


    def test_3dw_tail_not_lost_before_4dw_request(self):
        # On a 128-bit datapath a 3DW header shifts the payload by 3 dwords, so a write whose
        # payload fills its last input beat owes one more output beat after that input beat has
        # been accepted. A 4DW request presented right behind it must not switch the 3DW/4DW header
        # inserter mux before that beat is out, or the write loses its tail and its `last`.
        data_width = 128
        dut = LitePCIeTLPPacketizer(
            data_width    = data_width,
            endianness    = "big",
            address_width = 64,
            capabilities  = ["REQUEST"],
        )
        requests = [
            # (we, adr, len) : MWr32 (3DW + 64 dwords = 17 beats), MRd64 (4DW = 1 beat), ...
            (1, 0x0000_1000,   64),
            (0, 0x1_0000_2000, 128),
            (1, 0x0000_1100,   64),
            (1, 0x1_0000_3000, 64),
            (1, 0x0000_1200,   64),
            (0, 0x0000_2200,   128),
        ]
        packets = []

        @passive
        def monitor():
            yield dut.source.ready.eq(1)
            while True:
                if (yield dut.source.valid) and (yield dut.source.ready):
                    first = (yield dut.source.first)
                    last  = (yield dut.source.last)
                    be    = (yield dut.source.be)
                    if first or not packets:
                        packets.append({"beats": 0, "dwords": 0, "first": first, "last": 0,
                                        "first_inside": False})
                    elif packets[-1]["last"]:
                        packets.append({"beats": 0, "dwords": 0, "first": first, "last": 0,
                                        "first_inside": False})
                    pkt = packets[-1]
                    pkt["beats"]  += 1
                    pkt["dwords"] += sum(((be >> (4*lane)) & 0xf) == 0xf for lane in range(4))
                    pkt["last"]    = last
                yield

        def stimulus():
            sink = dut.req_sink
            for we, adr, length in requests:
                nbeats = (length + 3)//4 if we else 1
                yield sink.we.eq(we)
                yield sink.adr.eq(adr)
                yield sink.len.eq(length)
                yield sink.req_id.eq(0x0100)
                yield sink.tag.eq(0x3)
                for beat in range(nbeats):
                    yield sink.dat.eq(0x1111_1111*(beat + 1))
                    yield sink.first.eq(beat == 0)
                    yield sink.last.eq(beat == nbeats - 1)
                    yield sink.valid.eq(1)
                    yield
                    while not (yield sink.ready):
                        yield
            yield sink.valid.eq(0)
            for _ in range(40):
                yield

        run_simulation(dut, [stimulus(), monitor()])

        self.assertEqual(len(packets), len(requests))
        for (we, adr, length), pkt in zip(requests, packets):
            header_dwords = 4 if adr >> 32 else 3
            with self.subTest(we=we, adr=hex(adr)):
                self.assertEqual(pkt["first"], 1)
                self.assertEqual(pkt["last"], 1)
                self.assertEqual(pkt["beats"], (header_dwords + (length if we else 0) + 3)//4)
                if we:
                    self.assertEqual(pkt["dwords"], header_dwords + length)


if __name__ == "__main__":
    unittest.main()
