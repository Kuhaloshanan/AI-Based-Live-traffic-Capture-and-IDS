"""
Live Network Packet Capture Module, Capture packet metadata from the selected network interface.
This is the first stage of our real-time intrusion detection system.
"""

from datetime import datetime

from scapy.all import sniff, IP, IPv6, TCP, UDP



#  NETWORK INTERFACE CONFIGURATION

INTERFACE_NAME = "Wi-Fi"



# EXTRACT INFORMATION FROM A NETWORK PACKET


def extract_packet_info(packet):
    """
    Extract useful metadata from a captured packet.

    We extract packet headers and size, not application payloads.
    """

    # Convert the packet timestamp into a readable date and time
    timestamp = datetime.fromtimestamp(
        float(packet.time)
    ).isoformat(timespec="milliseconds")

    # Default values in case the packet has no IP layer
    source_ip = "N/A"
    destination_ip = "N/A"

    # Identify IPv4 packets
    if IP in packet:
        source_ip = packet[IP].src
        destination_ip = packet[IP].dst

    # Identify IPv6 packets
    elif IPv6 in packet:
        source_ip = packet[IPv6].src
        destination_ip = packet[IPv6].dst

    # Default transport information
    protocol = "OTHER"
    source_port = None
    destination_port = None
    tcp_flags = None

    # Extract TCP information
    if TCP in packet:
        protocol = "TCP"
        source_port = packet[TCP].sport
        destination_port = packet[TCP].dport
        tcp_flags = str(packet[TCP].flags)

    # Extract UDP information
    elif UDP in packet:
        protocol = "UDP"
        source_port = packet[UDP].sport
        destination_port = packet[UDP].dport

    # Calculate the complete packet size in bytes
    packet_size = len(packet)

    # Return the extracted information as a dictionary
    return {
        "timestamp": timestamp,
        "source_ip": source_ip,
        "destination_ip": destination_ip,
        "protocol": protocol,
        "source_port": source_port,
        "destination_port": destination_port,
        "packet_size": packet_size,
        "tcp_flags": tcp_flags
    }



# 3. DISPLAY CAPTURED PACKET INFORMATION


def display_packet(packet):
    """
    Display metadata for each captured packet.
    """

    # Extract the packet information
    info = extract_packet_info(packet)

    # Format source and destination endpoints
    source = f"{info['source_ip']}:{info['source_port'] or '-'}"
    destination = (
        f"{info['destination_ip']}:{info['destination_port'] or '-'}"
    )

    # Display the captured packet information
    print(
        f"[{info['timestamp']}] "
        f"{info['protocol']} | "
        f"{source} -> {destination} | "
        f"Size: {info['packet_size']} bytes | "
        f"TCP Flags: {info['tcp_flags'] or '-'}"
    )



# START LIVE PACKET CAPTURE


def start_capture():
    """
    Continuously capture packets from the selected interface.

    Press CTRL + C to stop packet capture.
    """

    print("=" * 65)
    print("AI CYBER THREAT INTELLIGENCE SYSTEM")
    print("LIVE NETWORK PACKET CAPTURE")
    print("=" * 65)

    print(f"Monitoring interface: {INTERFACE_NAME}")
    print("Waiting for network packets...")
    print("Press CTRL + C to stop monitoring.")
    print("-" * 65)

    try:
        # Start capturing packets
        # store=False avoids keeping the complete packet list in memory
        sniff(
            iface=INTERFACE_NAME,
            prn=display_packet,
            store=False
        )

    except KeyboardInterrupt:
        print("\nPacket capture stopped by user.")

    print("Live capture session ended.")



#PROGRAM ENTRY POINT

if __name__ == "__main__":
    start_capture()
