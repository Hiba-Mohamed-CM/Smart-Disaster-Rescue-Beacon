import numpy as np
from collections import deque

# =========================================================
# SMART DISASTER RESCUE BEACON
# LoRa + Mesh Communication Simulation
# =========================================================

RNG = np.random.default_rng(42)

COMMUNICATION_RANGE = 2.5
MAX_HOPS = 6
PACKET_SUCCESS_RATE = 0.95

N_ALERTS = 1000


# ---------------------------------------------------------
# Fixed beacon topology
# ---------------------------------------------------------
#
# The positions are intentionally arranged so that:
#
# Base Station
#       |
#   Beacon 0
#       |
#   Beacon 1
#    /       \
# Beacon 2  Beacon 3
#    |          |
# Beacon 4   Beacon 5
#    |          |
# Beacon 6   Beacon 7
#
# Additional beacons are placed around the network.
#
# This allows genuine multi-hop testing.
# ---------------------------------------------------------

positions = np.array([
    [0.0, 0.0],     # 0
    [2.0, 0.0],     # 1
    [4.0, 0.0],     # 2
    [4.0, 2.0],     # 3
    [6.0, 0.0],     # 4
    [6.0, 2.0],     # 5
    [8.0, 0.0],     # 6
    [8.0, 2.0],     # 7
    [2.0, 2.0],     # 8
    [2.0, 4.0],     # 9
    [4.0, 4.0],     # 10
    [6.0, 4.0],     # 11
    [8.0, 4.0],     # 12
    [4.0, 6.0],     # 13
    [6.0, 6.0],     # 14
    [8.0, 6.0],     # 15
    [10.0, 2.0],    # 16
    [10.0, 4.0],    # 17
    [10.0, 6.0],    # 18
    [10.0, 8.0],    # 19
])

BASE_STATION = np.array([0.0, -2.0])


# ---------------------------------------------------------
# Distance function
# ---------------------------------------------------------

def distance(a, b):
    return np.linalg.norm(a - b)


# ---------------------------------------------------------
# Build mesh graph
# ---------------------------------------------------------

graph = {
    i: []
    for i in range(len(positions))
}


for i in range(len(positions)):

    for j in range(len(positions)):

        if i == j:
            continue

        d = distance(
            positions[i],
            positions[j]
        )

        if d <= COMMUNICATION_RANGE:

            graph[i].append(j)


# ---------------------------------------------------------
# Find beacons directly connected to base station
# ---------------------------------------------------------

base_neighbors = []

for i in range(len(positions)):

    d = distance(
        positions[i],
        BASE_STATION
    )

    if d <= COMMUNICATION_RANGE:

        base_neighbors.append(i)


# ---------------------------------------------------------
# Find shortest route to base station
# ---------------------------------------------------------

def find_route(source):

    queue = deque()

    queue.append(
        (source, [source])
    )

    visited = set()

    while queue:

        current, path = queue.popleft()

        if current in visited:
            continue

        visited.add(current)

        # Current beacon can reach base station
        if current in base_neighbors:

            return path

        # Maximum relay hops
        if len(path) - 1 >= MAX_HOPS:

            continue

        for neighbor in graph[current]:

            if neighbor not in visited:

                queue.append(
                    (
                        neighbor,
                        path + [neighbor]
                    )
                )

    return None


# ---------------------------------------------------------
# Simulate transmission over route
# ---------------------------------------------------------

def transmit_route(route):

    if route is None:

        return False

    # Beacon-to-beacon transmissions
    for i in range(len(route) - 1):

        if RNG.random() > PACKET_SUCCESS_RATE:

            return False

    # Final beacon to base station
    if RNG.random() > PACKET_SUCCESS_RATE:

        return False

    return True


# =========================================================
# DISPLAY NETWORK
# =========================================================

print("=" * 75)
print("SMART DISASTER RESCUE BEACON")
print("LoRa + MESH COMMUNICATION SIMULATION")
print("=" * 75)

print()

print(
    f"Number of beacons       : {len(positions)}"
)

print(
    f"Communication range     : "
    f"{COMMUNICATION_RANGE:.2f} m"
)

print(
    f"Maximum relay hops      : {MAX_HOPS}"
)

print(
    f"Packet success rate     : "
    f"{PACKET_SUCCESS_RATE:.2f}"
)


print()
print("=" * 75)
print("BEACON NETWORK")
print("=" * 75)


for i in range(len(positions)):

    print(
        f"Beacon {i:02d} | "
        f"Position = "
        f"({positions[i,0]:5.2f}, "
        f"{positions[i,1]:5.2f}) | "
        f"Neighbors = {len(graph[i])}"
    )


print()

print(
    "Beacons directly reachable from "
    "base station:",
    base_neighbors
)


# =========================================================
# TEST ROUTES
# =========================================================

print()
print("=" * 75)
print("ROUTE TEST")
print("=" * 75)

test_sources = [
    0,
    3,
    7,
    12,
    15,
    19
]

for source in test_sources:

    route = find_route(source)

    if route is None:

        print(
            f"Beacon {source:02d} -> "
            f"NO ROUTE"
        )

    else:

        print(
            f"Beacon {source:02d} -> "
            f"Base Station | "
            f"Route: {route} | "
            f"Hops: {len(route)}"
        )


# =========================================================
# SURVIVOR ALERT SIMULATION
# =========================================================

successful = 0
failed_no_route = 0
failed_transmission = 0

hop_counts = []

for _ in range(N_ALERTS):

    source = RNG.integers(
        0,
        len(positions)
    )

    route = find_route(source)

    if route is None:

        failed_no_route += 1

        continue

    delivered = transmit_route(route)

    if delivered:

        successful += 1

        hop_counts.append(
            len(route)
        )

    else:

        failed_transmission += 1


# =========================================================
# RESULTS
# =========================================================

delivery_rate = (
    successful /
    N_ALERTS *
    100
)

route_failure_rate = (
    failed_no_route /
    N_ALERTS *
    100
)

transmission_failure_rate = (
    failed_transmission /
    N_ALERTS *
    100
)


print()
print("=" * 75)
print("SURVIVOR ALERT SIMULATION RESULTS")
print("=" * 75)

print(
    f"Total alerts              : "
    f"{N_ALERTS}"
)

print(
    f"Successfully delivered    : "
    f"{successful}"
)

print(
    f"No available route        : "
    f"{failed_no_route}"
)

print(
    f"Transmission failures     : "
    f"{failed_transmission}"
)

print(
    f"Delivery rate             : "
    f"{delivery_rate:.2f}%"
)

print(
    f"Route failure rate        : "
    f"{route_failure_rate:.2f}%"
)

print(
    f"Transmission failure rate : "
    f"{transmission_failure_rate:.2f}%"
)


if hop_counts:

    print(
        f"Average route length     : "
        f"{np.mean(hop_counts):.2f} beacons"
    )

    print(
        f"Maximum route length     : "
        f"{max(hop_counts)} beacons"
    )


print()
print("=" * 75)
print("SIMULATION COMPLETE")
print("=" * 75)