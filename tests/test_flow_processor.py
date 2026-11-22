
"""
Test Network Flow Processing

Creates synthetic TCP packets to verify:
1. Bidirectional flow identification.
2. Packet counting.
3. Forward and reverse traffic counting.
4. Total byte calculation.
5. Flow duration.
6. Inactivity timeout.
"""

import unittest

from scapy.all import IP, TCP, Raw

from src.flow_processor import FlowProcessor


def create_test_packet(
    source_ip,
    destination_ip,
    source_port,
    destination_port,
    timestamp,
    payload
):
    """
    Create a synthetic TCP packet with a controlled timestamp.
    """

    packet = (
        IP(src=source_ip, dst=destination_ip)
        / TCP(sport=source_port, dport=destination_port)
        / Raw(load=payload)
    )

    # Assign a controlled timestamp for predictable testing.
    packet.time = timestamp

    return packet


class TestFlowProcessor(unittest.TestCase):

    def test_bidirectional_flow(self):

        processor = FlowProcessor(idle_timeout=30)

        # Create three packets belonging to one communication.
        packets = [
            create_test_packet(
                "192.0.2.10", "198.51.100.20",
                5000, 443, 100.0, b"A" * 10
            ),

            # Reverse direction.
            create_test_packet(
                "198.51.100.20", "192.0.2.10",
                443, 5000, 101.0, b"B" * 20
            ),

            create_test_packet(
                "192.0.2.10", "198.51.100.20",
                5000, 443, 102.0, b"C" * 30
            )
        ]

        # Process every packet.
        for packet in packets:
            processor.process_packet(packet)

        # Complete the flow and retrieve its statistics.
        completed_flows = processor.flush()

        self.assertEqual(len(completed_flows), 1)

        flow = completed_flows[0]

        # Verify packet counts.
        self.assertEqual(flow["packet_count"], 3)
        self.assertEqual(flow["forward_packets"], 2)
        self.assertEqual(flow["reverse_packets"], 1)

        # Verify total bytes against actual synthetic packet sizes.
        expected_bytes = sum(len(packet) for packet in packets)

        self.assertEqual(flow["total_bytes"], expected_bytes)

        # Verify the duration.
        self.assertAlmostEqual(
            flow["duration_seconds"], 2.0
        )

        # Verify average packet size.
        self.assertAlmostEqual(
            flow["average_packet_size"],
            expected_bytes / 3
        )

    def test_idle_timeout(self):

        processor = FlowProcessor(idle_timeout=30)

        # First packet starts a flow at timestamp 100.
        first_packet = create_test_packet(
            "192.0.2.10", "198.51.100.20",
            5000, 443, 100.0, b"FIRST"
        )

        processor.process_packet(first_packet)

        # Another packet arrives after 31 seconds.
        # The original flow should expire.
        second_packet = create_test_packet(
            "192.0.2.10", "198.51.100.20",
            5000, 443, 131.0, b"SECOND"
        )

        completed_flows = processor.process_packet(second_packet)

        self.assertEqual(len(completed_flows), 1)

        self.assertEqual(
            completed_flows[0]["packet_count"], 1
        )

        # The second packet should belong to a new flow.
        remaining_flows = processor.flush()

        self.assertEqual(len(remaining_flows), 1)
        self.assertEqual(
            remaining_flows[0]["packet_count"], 1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
