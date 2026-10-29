import numpy as np
from collections import deque

# =========================================================
# SMART DISASTER RESCUE BEACON
# Mesh Communication Range Sensitivity Analysis
# =========================================================

RNG_SEED = 42

N_ALERTS = 1000
MAX_HOPS = 6
PACKET_SUCCESS_RATE = 0.95

COMMUNICATION_RANGES = [
    1.5,
    2.0,
    2.5,
    3.0,
    3.5
]

# Same topology used in the previous simulation
positions = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [4.0, 0.0],
    [4.0, 2.0],
    [6.0, 0.0],
    [6.0, 2.0],
    [8.0, 0.0],
    [8.0, 2.0],
    [2.0, 2.0],
    [2.0, 4.0],
    [4.0, 4.0],
    [6.0, 4.0],
    [8.0, 4.0],
    [4.0, 6.0],
    [6.0, 6.0],
    [8.0, 6.0],
    [10.0, 2.0],
    [10.0, 4.0],
    [10.0, 6.0],
    [10.0, 8.0],
])

BASE_STATION = np.array([0.0, -2.0])


def distance(a, b):
    return np.linalg.norm(a - b)


def build_graph(comm_range):

    graph = {
        i: []
        for i in range(len(positions))
    }

    for i in range(len(positions)):

        for j in range(len(positions)):

            if i == j:
                continue

            if distance(
                positions[i],
                positions[j]
            ) <= comm_range:

                graph[i].append(j)

    base_neighbors = []

    for i in range(len(positions)):

        if distance(
            positions[i],
            BASE_STATION
        ) <= comm_range:

            base_neighbors.append(i)

    return graph, base_neighbors


def find_route(
    source,
    graph,
    base_neighbors
):

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

        if current in base_neighbors:

            return path

        # Actual relay hops:
        # number of beacon-to-beacon links
        relay_hops = len(path) - 1

        if relay_hops >= MAX_HOPS:
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


def run_simulation(comm_range, seed):

    rng = np.random.default_rng(seed)

    graph, base_neighbors = build_graph(
        comm_range
    )

    successful = 0
    no_route = 0
    transmission_failure = 0

    hop_counts = []

    for _ in range(N_ALERTS):

        source = rng.integers(
            0,
            len(positions)
        )

        route = find_route(
            source,
            graph,
            base_neighbors
        )

        if route is None:

            no_route += 1
            continue

        # Number of beacon-to-beacon relay hops
        relay_hops = len(route) - 1

        # Include final beacon -> base transmission
        total_transmissions = relay_hops + 1

        delivered = True

        for _ in range(total_transmissions):

            if (
                rng.random()
                > PACKET_SUCCESS_RATE
            ):

                delivered = False
                break

        if delivered:

            successful += 1
            hop_counts.append(
                total_transmissions
            )

        else:

            transmission_failure += 1

    delivery_rate = (
        successful /
        N_ALERTS *
        100
    )

    route_rate = (
        (N_ALERTS - no_route) /
        N_ALERTS *
        100
    )

    transmission_failure_rate = (
        transmission_failure /
        N_ALERTS *
        100
    )

    if hop_counts:

        average_hops = np.mean(
            hop_counts
        )

    else:

        average_hops = 0

    return {
        "range": comm_range,
        "base_neighbors": len(base_neighbors),
        "delivery_rate": delivery_rate,
        "route_rate": route_rate,
        "transmission_failure_rate":
            transmission_failure_rate,
        "average_hops": average_hops
    }


# =========================================================
# Run experiment
# =========================================================

print("=" * 78)
print("SMART DISASTER RESCUE BEACON")
print("MESH COMMUNICATION RANGE SENSITIVITY ANALYSIS")
print("=" * 78)

print()
print(f"Alerts per range       : {N_ALERTS}")
print(f"Maximum relay hops     : {MAX_HOPS}")
print(f"Packet success rate    : {PACKET_SUCCESS_RATE:.2f}")

print()
print("=" * 78)

results = []

for comm_range in COMMUNICATION_RANGES:

    result = run_simulation(
        comm_range,
        RNG_SEED
    )

    results.append(result)

    print()
    print(
        f"Communication range: "
        f"{comm_range:.1f} m"
    )

    print(
        f"Base-connected beacons : "
        f"{result['base_neighbors']}"
    )

    print(
        f"Route availability     : "
        f"{result['route_rate']:.2f}%"
    )

    print(
        f"Delivery rate          : "
        f"{result['delivery_rate']:.2f}%"
    )

    print(
        f"Transmission failures  : "
        f"{result['transmission_failure_rate']:.2f}%"
    )

    print(
        f"Average transmissions  : "
        f"{result['average_hops']:.2f}"
    )


# =========================================================
# Summary table
# =========================================================

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(
    f"{'Range':>8} | "
    f"{'Base':>5} | "
    f"{'Routes':>9} | "
    f"{'Delivery':>10} | "
    f"{'Avg Hops':>9}"
)

print("-" * 78)

for r in results:

    print(
        f"{r['range']:>7.1f} m | "
        f"{r['base_neighbors']:>5} | "
        f"{r['route_rate']:>8.2f}% | "
        f"{r['delivery_rate']:>9.2f}% | "
        f"{r['average_hops']:>9.2f}"
    )


print()
print("=" * 78)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 78)