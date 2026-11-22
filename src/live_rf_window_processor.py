
"""
Live Random Forest Packet Window Processor.

Captures IPv4 packets and groups them into non-overlapping
10-packet and 100-packet windows.

Each completed window contains:
1. A shared network flow identifier.
2. The packet window size.
3. The 39 features expected by the Random Forest model.

NOTE:
The live feature definitions are candidate representations
and should not be assumed to perfectly reproduce the
original CICIoT2023 feature extraction process.
"""

import math
from collections import Counter

from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import LLC


# =========================================================
# RANDOM FOREST FEATURE SCHEMA
# =========================================================

# Exact feature names and order expected by the trained model.
RF_FEATURE_COLUMNS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance"
]


# =========================================================
# LIVE RANDOM FOREST WINDOW PROCESSOR
# =========================================================

class LiveRFWindowProcessor:

    def __init__(self, window_sizes=(10, 100), idle_timeout=30):
        """
        Initialize the packet-window processor.

        window_sizes:
            Packet counts required to complete a window.

        idle_timeout:
            Maximum inactivity period before incomplete
            packet windows are discarded.
        """

        self.window_sizes = tuple(window_sizes)
        self.idle_timeout = idle_timeout

        # Each network conversation maintains its own buffers.
        self.active_windows = {}

    # =====================================================
    # EXTRACT PACKET INFORMATION
    # =====================================================

    def extract_packet(self, packet):
        """
        Extract information required for feature calculation.

        The Random Forest live pipeline currently accepts
        IPv4 packets only.
        """

        # Ignore packets without IPv4.
        if IP not in packet:
            return None

        layer = packet[IP]

        src = layer.src
        dst = layer.dst

        ttl = int(layer.ttl or 0)
        protocol_type = int(layer.proto)

        transport = "OTHER"

        sport = None
        dport = None

        # -------------------------------------------------
        # Identify transport protocol.
        # -------------------------------------------------

        if TCP in packet:

            transport = "TCP"

            sport = int(packet[TCP].sport)
            dport = int(packet[TCP].dport)

            protocol_type = 6

        elif UDP in packet:

            transport = "UDP"

            sport = int(packet[UDP].sport)
            dport = int(packet[UDP].dport)

            protocol_type = 17

        elif ICMP in packet:

            transport = "ICMP"

            protocol_type = 1

        # IPv4 protocol number 2 represents IGMP.
        elif int(layer.proto) == 2:

            transport = "IGMP"

            protocol_type = 2

        # -------------------------------------------------
        # Calculate packet header length.
        # -------------------------------------------------

        header_length = int(layer.ihl or 5) * 4

        if TCP in packet:

            header_length += int(
                packet[TCP].dataofs or 5
            ) * 4

        elif UDP in packet:

            header_length += 8

        elif ICMP in packet:

            header_length += 8

        # -------------------------------------------------
        # Extract TCP flags.
        # -------------------------------------------------

        flags = (
            int(packet[TCP].flags)
            if TCP in packet
            else 0
        )

        # Collect source and destination ports.
        ports = {sport, dport}

        # -------------------------------------------------
        # Return extracted packet metadata.
        # -------------------------------------------------

        return {

            "src": src,
            "dst": dst,

            "sport": sport,
            "dport": dport,

            "transport": transport,

            "timestamp": float(packet.time),

            "packet_size": len(packet),

            "header_length": header_length,

            "ttl": ttl,

            "protocol_type": protocol_type,

            "flags": flags,

            # TCP flag indicators.
            "fin": bool(flags & 0x01),
            "syn": bool(flags & 0x02),
            "rst": bool(flags & 0x04),
            "psh": bool(flags & 0x08),
            "ack": bool(flags & 0x10),
            "ece": bool(flags & 0x40),
            "cwr": bool(flags & 0x80),

            # Application protocol indicators.
            "http": 80 in ports,
            "https": 443 in ports,
            "dns": 53 in ports,
            "telnet": 23 in ports,

            "smtp": bool(
                ports.intersection({25, 465, 587})
            ),

            "ssh": 22 in ports,

            "irc": bool(
                ports.intersection({
                    194, 6667, 6668, 6669
                })
            ),

            "dhcp": bool(
                ports.intersection({67, 68})
            ),

            # Network protocol indicators.
            "tcp": transport == "TCP",
            "udp": transport == "UDP",

            "arp": False,

            "icmp": transport == "ICMP",

            "igmp": transport == "IGMP",

            "ipv": True,

            "llc": LLC in packet
        }

    # =====================================================
    # CREATE BIDIRECTIONAL FLOW IDENTIFIER
    # =====================================================

    def create_flow_key(self, metadata):
        """
        Create a bidirectional flow identifier.

        The same identifier format is used by FlowProcessor.

        Sorting the endpoints ensures that packets travelling
        in either direction belong to the same conversation.
        """

        endpoint_a = (
            metadata["src"],
            metadata["sport"]
            if metadata["sport"] is not None
            else -1
        )

        endpoint_b = (
            metadata["dst"],
            metadata["dport"]
            if metadata["dport"] is not None
            else -1
        )

        endpoint_a, endpoint_b = sorted(
            [endpoint_a, endpoint_b]
        )

        # FlowProcessor currently identifies unsupported
        # IPv4 protocols, including IGMP, as OTHER.
        # Use the same key protocol for correlation.
        key_protocol = metadata["transport"]

        if key_protocol == "IGMP":
            key_protocol = "OTHER"

        return (
            key_protocol,
            endpoint_a,
            endpoint_b
        )

    # =====================================================
    # BUILD RANDOM FOREST FEATURES
    # =====================================================

    def build_features(self, packets):
        """
        Calculate the 39 candidate features from one window.
        """

        count = len(packets)

        sizes = [
            packet["packet_size"]
            for packet in packets
        ]

        timestamps = [
            packet["timestamp"]
            for packet in packets
        ]

        # Time difference between first and last packets.
        duration = max(
            0.0,
            timestamps[-1] - timestamps[0]
        )

        # Calculate inter-arrival times.
        intervals = [
            max(
                0.0,
                timestamps[i] - timestamps[i - 1]
            )
            for i in range(1, count)
        ]

        total_bytes = sum(sizes)

        average_size = total_bytes / count

        # Population variance.
        variance = sum(
            (size - average_size) ** 2
            for size in sizes
        ) / count

        std = math.sqrt(variance)

        # -------------------------------------------------
        # Count TCP flags.
        # -------------------------------------------------

        fin_count = sum(
            packet["fin"] for packet in packets
        )

        syn_count = sum(
            packet["syn"] for packet in packets
        )

        rst_count = sum(
            packet["rst"] for packet in packets
        )

        psh_count = sum(
            packet["psh"] for packet in packets
        )

        ack_count = sum(
            packet["ack"] for packet in packets
        )

        ece_count = sum(
            packet["ece"] for packet in packets
        )

        cwr_count = sum(
            packet["cwr"] for packet in packets
        )

        # -------------------------------------------------
        # Calculate protocol indicators.
        # -------------------------------------------------

        def indicator(name):
            return int(
                any(packet[name] for packet in packets)
            )

        # -------------------------------------------------
        # Construct all 39 model features.
        # -------------------------------------------------

        features = {

            "Header_Length": (
                sum(
                    packet["header_length"]
                    for packet in packets
                ) / count
            ),

            "Protocol Type": (
                sum(
                    packet["protocol_type"]
                    for packet in packets
                ) / count
            ),

            "Time_To_Live": (
                sum(
                    packet["ttl"]
                    for packet in packets
                ) / count
            ),

            "Rate": (
                count / duration
                if duration > 0
                else 0
            ),

            # TCP flag ratios.
            "fin_flag_number": fin_count / count,
            "syn_flag_number": syn_count / count,
            "rst_flag_number": rst_count / count,
            "psh_flag_number": psh_count / count,
            "ack_flag_number": ack_count / count,
            "ece_flag_number": ece_count / count,
            "cwr_flag_number": cwr_count / count,

            # TCP flag counts.
            "ack_count": ack_count,
            "syn_count": syn_count,
            "fin_count": fin_count,
            "rst_count": rst_count,

            # Application protocol indicators.
            "HTTP": indicator("http"),
            "HTTPS": indicator("https"),
            "DNS": indicator("dns"),
            "Telnet": indicator("telnet"),
            "SMTP": indicator("smtp"),
            "SSH": indicator("ssh"),
            "IRC": indicator("irc"),

            # Network protocol indicators.
            "TCP": indicator("tcp"),
            "UDP": indicator("udp"),
            "DHCP": indicator("dhcp"),
            "ARP": indicator("arp"),
            "ICMP": indicator("icmp"),
            "IGMP": indicator("igmp"),
            "IPv": indicator("ipv"),
            "LLC": indicator("llc"),

            # Packet size statistics.
            "Tot sum": total_bytes,
            "Min": min(sizes),
            "Max": max(sizes),
            "AVG": average_size,
            "Std": std,
            "Tot size": average_size,

            # Average packet inter-arrival time.
            "IAT": (
                sum(intervals) / len(intervals)
                if intervals
                else 0
            ),

            # Packet window size.
            "Number": count,

            "Variance": variance
        }

        # Preserve the exact feature order expected by
        # the trained Random Forest model.
        return {
            name: features[name]
            for name in RF_FEATURE_COLUMNS
        }

    # =====================================================
    # PROCESS INCOMING PACKETS
    # =====================================================

    def process_packet(self, packet):
        """
        Add a packet to its corresponding network conversation.

        Returns completed windows containing:

        {
            "flow_key": shared flow identifier,
            "window_size": 10 or 100,
            "features": dictionary of 39 model features
        }
        """

        # Step 1: Extract packet information.
        metadata = self.extract_packet(packet)

        if metadata is None:
            return []

        timestamp = metadata["timestamp"]

        # Step 2: Remove incomplete windows belonging to
        # conversations that have exceeded the idle timeout.
        for old_key, state in list(
            self.active_windows.items()
        ):

            if (
                timestamp - state["last_seen"]
                >= self.idle_timeout
            ):

                del self.active_windows[old_key]

        # Step 3: Generate the shared bidirectional flow key.
        key = self.create_flow_key(metadata)

        # Step 4: Initialize buffers for a new conversation.
        if key not in self.active_windows:

            self.active_windows[key] = {

                "last_seen": timestamp,

                "buffers": {
                    size: []
                    for size in self.window_sizes
                }
            }

        state = self.active_windows[key]

        state["last_seen"] = timestamp

        # Store all completed windows from this packet.
        completed = []

        # Step 5: Process each configured window size.
        for size in self.window_sizes:

            buffer = state["buffers"][size]

            buffer.append(metadata)

            # A window is ready when it reaches its target size.
            if len(buffer) == size:

                # Calculate the 39 model input features.
                features = self.build_features(buffer)

                # Attach the shared flow identifier.
                # This allows the unified decision engine
                # to associate this prediction with its flow.
                completed.append({

                    "flow_key": key,

                    "window_size": size,

                    "features": features
                })

                # Start a new non-overlapping window.
                state["buffers"][size] = []

        return completed


# =========================================================
# STANDALONE PROCESSOR TEST
# =========================================================

if __name__ == "__main__":

    from scapy.layers.l2 import Ether

    print(
        "Testing live RF packet window processor..."
    )

    processor = LiveRFWindowProcessor()

    results = []

    # Generate 100 synthetic TCP packets for testing.
    for i in range(100):

        packet = (
            Ether() /
            IP(
                src="192.168.1.10",
                dst="192.168.1.20",
                ttl=64
            ) /
            TCP(
                sport=50000,
                dport=443,
                flags="PA"
            ) /
            (b"TEST" * 10)
        )

        packet.time = 1000 + i * 0.01

        results.extend(
            processor.process_packet(packet)
        )

    # Count completed windows by size.
    counts = Counter(
        result["window_size"]
        for result in results
    )

    print(
        "Completed windows:",
        dict(counts)
    )

    # Verify the feature schema and flow identifier.
    for result in results:

        assert len(result["features"]) == 39

        assert (
            list(result["features"].keys())
            == RF_FEATURE_COLUMNS
        )

        assert "flow_key" in result

    print("Feature count: 39")
    print("Flow identifiers: verified")

    print(
        "All feature records have the expected schema."
    )
