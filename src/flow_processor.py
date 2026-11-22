
"""
Network Flow Processing Module
Group individual network packets into bidirectional network flows.
This is a preliminary flow processor. Its statistics will be
validated against CICIoT2023 feature definitions before being
used as input to the trained Random Forest model.
"""

import math
from datetime import datetime

from scapy.all import IP, IPv6, TCP, UDP


class FlowProcessor:

    def __init__(self, idle_timeout=30):
        """
        Initialize the flow processor.

        idle_timeout:
            Number of seconds a flow can remain inactive
            before it is considered complete.
        """

        self.idle_timeout = idle_timeout

        # Dictionary containing currently active network flows.
        self.active_flows = {}

    #  EXTRACT PACKET INFORMATION

    def extract_packet_metadata(self, packet):
        """
        Extract network and transport information from a packet.

        We only inspect packet headers and packet size.
        """

        source_ip = None
        destination_ip = None

        source_port = None
        destination_port = None

        protocol = "OTHER"

        # Extract IPv4 information.
        if IP in packet:

            source_ip = packet[IP].src
            destination_ip = packet[IP].dst

        # Extract IPv6 information.
        elif IPv6 in packet:

            source_ip = packet[IPv6].src
            destination_ip = packet[IPv6].dst

        # Ignore packets without an IP layer for now.
        else:
            return None

        # Identify TCP traffic.
        if TCP in packet:

            protocol = "TCP"

            source_port = int(packet[TCP].sport)
            destination_port = int(packet[TCP].dport)

        # Identify UDP traffic.
        elif UDP in packet:

            protocol = "UDP"

            source_port = int(packet[UDP].sport)
            destination_port = int(packet[UDP].dport)

        # Identify ICMP traffic.
        elif IP in packet and packet[IP].proto == 1:

            protocol = "ICMP"

        elif IPv6 in packet and packet[IPv6].nh == 58:

            protocol = "ICMPv6"

        # Measure packet size in bytes.
        packet_size = len(packet)

        # Obtain the packet timestamp.
        timestamp = float(packet.time)

        return {
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "source_port": source_port,
            "destination_port": destination_port,
            "protocol": protocol,
            "packet_size": packet_size,
            "timestamp": timestamp
        }

    # CREATE A BIDIRECTIONAL FLOW IDENTIFIER

    def create_flow_key(self, metadata):
        """
        Create a flow identifier using the 5-tuple.

        Sorting the endpoints allows packets travelling
        in either direction to belong to the same flow.
        """

        source_endpoint = (
            metadata["source_ip"],
            metadata["source_port"]
            if metadata["source_port"] is not None else -1
        )

        destination_endpoint = (
            metadata["destination_ip"],
            metadata["destination_port"]
            if metadata["destination_port"] is not None else -1
        )

        # Sort endpoints to make the flow bidirectional.
        endpoint_a, endpoint_b = sorted(
            [source_endpoint, destination_endpoint]
        )

        return (
            metadata["protocol"],
            endpoint_a,
            endpoint_b
        )


    #  CREATE A NEW NETWORK FLOW


    def create_new_flow(self, metadata):
        """
        Initialize the statistics for a new flow.
        """

        return {
            "source_ip": metadata["source_ip"],
            "destination_ip": metadata["destination_ip"],

            "source_port": metadata["source_port"],
            "destination_port": metadata["destination_port"],

            "protocol": metadata["protocol"],

            "first_seen": metadata["timestamp"],
            "last_seen": metadata["timestamp"],

            "packet_count": 0,
            "total_bytes": 0,

            "forward_packets": 0,
            "reverse_packets": 0,

            "forward_bytes": 0,
            "reverse_bytes": 0,

            "min_packet_size": float("inf"),
            "max_packet_size": 0,

            "sum_packet_sizes": 0,
            "sum_squared_packet_sizes": 0,

            # The first packet defines the forward direction.
            "initiator_endpoint": (
                metadata["source_ip"],
                metadata["source_port"]
                if metadata["source_port"] is not None else -1
            )
        }


    #  UPDATE FLOW STATISTICS


    def update_flow(self, flow, metadata):
        """
        Update the statistics when another packet
        belongs to an existing flow.
        """

        packet_size = metadata["packet_size"]

        source_endpoint = (
            metadata["source_ip"],
            metadata["source_port"]
            if metadata["source_port"] is not None else -1
        )

        # Update packet and byte totals.
        flow["packet_count"] += 1
        flow["total_bytes"] += packet_size

        # Update forward or reverse direction statistics.
        if source_endpoint == flow["initiator_endpoint"]:

            flow["forward_packets"] += 1
            flow["forward_bytes"] += packet_size

        else:

            flow["reverse_packets"] += 1
            flow["reverse_bytes"] += packet_size

        # Update packet size statistics.
        flow["min_packet_size"] = min(
            flow["min_packet_size"],
            packet_size
        )

        flow["max_packet_size"] = max(
            flow["max_packet_size"],
            packet_size
        )

        flow["sum_packet_sizes"] += packet_size

        flow["sum_squared_packet_sizes"] += (
            packet_size ** 2
        )

        # Update the last packet timestamp.
        flow["last_seen"] = metadata["timestamp"]


    # 5FINALIZE A COMPLETED FLOW


    def finalize_flow(self, flow):
        """
        Calculate summary statistics for a completed flow.
        """

        packet_count = flow["packet_count"]

        duration = (
            flow["last_seen"] - flow["first_seen"]
        )

        mean_packet_size = (
            flow["sum_packet_sizes"] / packet_count
            if packet_count > 0 else 0
        )

        # Calculate population variance.
        variance = (
            flow["sum_squared_packet_sizes"] / packet_count
            - mean_packet_size ** 2
            if packet_count > 0 else 0
        )

        # Avoid tiny negative values caused by floating-point rounding.
        variance = max(0, variance)

        standard_deviation = math.sqrt(variance)

        # Calculate basic packet and byte rates.
        packet_rate = (
            packet_count / duration
            if duration > 0 else 0
        )

        byte_rate = (
            flow["total_bytes"] / duration
            if duration > 0 else 0
        )

        # Convert timestamps into readable values.
        start_time = datetime.fromtimestamp(
            flow["first_seen"]
        ).isoformat(timespec="milliseconds")

        end_time = datetime.fromtimestamp(
            flow["last_seen"]
        ).isoformat(timespec="milliseconds")

# Reconstruct the same bidirectional flow identifier
# used by the Random Forest packet-window processor.

        source_endpoint = (
            flow["source_ip"],
            flow["source_port"]
            if flow["source_port"] is not None else -1
        )

        destination_endpoint = (
            flow["destination_ip"],
            flow["destination_port"]
            if flow["destination_port"] is not None else -1
        )

        endpoint_a, endpoint_b = sorted(
            [source_endpoint, destination_endpoint]
        )

        flow_key = (
            flow["protocol"],
            endpoint_a,
            endpoint_b
        )
        # Return the completed flow information.
        return {

            "flow_key": flow_key,

            "source_ip": flow["source_ip"],
            "destination_ip": flow["destination_ip"],

            "source_port": flow["source_port"],
            "destination_port": flow["destination_port"],

            "protocol": flow["protocol"],

            "start_time": start_time,
            "end_time": end_time,

            "duration_seconds": duration,

            "packet_count": packet_count,
            "total_bytes": flow["total_bytes"],

            "forward_packets": flow["forward_packets"],
            "reverse_packets": flow["reverse_packets"],

            "forward_bytes": flow["forward_bytes"],
            "reverse_bytes": flow["reverse_bytes"],

            "min_packet_size": flow["min_packet_size"],
            "max_packet_size": flow["max_packet_size"],

            "average_packet_size": mean_packet_size,
            "standard_deviation": standard_deviation,
            "variance": variance,

            "packet_rate": packet_rate,
            "byte_rate": byte_rate
        }

    #  EXPIRE INACTIVE FLOWS


    def expire_flows(self, current_time):
        """
        Find flows that have exceeded the inactivity timeout.

        Completed flows are removed from active memory
        and returned to the caller.
        """

        completed_flows = []

        for flow_key, flow in list(self.active_flows.items()):

            inactive_time = (
                current_time - flow["last_seen"]
            )

            if inactive_time >= self.idle_timeout:

                completed_flow = self.finalize_flow(flow)

                completed_flows.append(completed_flow)

                del self.active_flows[flow_key]

        return completed_flows


    #  PROCESS AN INCOMING PACKET

    def process_packet(self, packet):
        """
        Process one incoming packet.

        Returns any flows that became complete.
        """

        metadata = self.extract_packet_metadata(packet)

        # Ignore unsupported packets.
        if metadata is None:
            return []

        # First expire flows that have become inactive.
        completed_flows = self.expire_flows(
            metadata["timestamp"]
        )

        # Identify the packet's network flow.
        flow_key = self.create_flow_key(metadata)

        # Create a flow if this communication is new.
        if flow_key not in self.active_flows:

            self.active_flows[flow_key] = (
                self.create_new_flow(metadata)
            )

        # Update the corresponding flow.
        flow = self.active_flows[flow_key]

        self.update_flow(flow, metadata)

        return completed_flows


    # FLUSH REMAINING FLOWS


    def flush(self):
        """
        Finalize all remaining active flows.

        Useful when stopping packet capture.
        """

        completed_flows = []

        for flow in self.active_flows.values():

            completed_flows.append(
                self.finalize_flow(flow)
            )

        self.active_flows.clear()

        return completed_flows
