import numpy as np
from collections import deque

# ============================================================
# SMART DISASTER RESCUE BEACON
# END-TO-END ROBUSTNESS SIMULATION
#
# Tests the effect of wireless packet reliability on
# end-to-end alert delivery.
# ============================================================

RNG_SEED = 42
N_ALERTS = 1000

# ------------------------------------------------------------
# Confidence fusion
# ------------------------------------------------------------

RF_WEIGHT = 0.50
SOUND_WEIGHT = 0.20
PIR_WEIGHT = 0.15
IR_WEIGHT = 0.15

CONFIDENCE_THRESHOLD = 0.50

# ------------------------------------------------------------
# Mesh parameters
# ------------------------------------------------------------

COMMUNICATION_RANGE = 3.0
MAX_RELAY_HOPS = 6

PACKET_SUCCESS_RATES = [
    0.80,
    0.85,
    0.90,
    0.95,
    0.98,
    1.00
]

BASE_STATION = np.array([0.0, -2.0])

# Same topology used in previous simulations
POSITIONS = np.array([
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


def distance(a, b):
    return np.linalg.norm(a - b)


# ============================================================
# BUILD MESH
# ============================================================

def build_mesh():

    graph = {
        i: []
        for i in range(len(POSITIONS))
    }

    for i in range(len(POSITIONS)):

        for j in range(len(POSITIONS)):

            if i == j:
                continue

            if distance(
                POSITIONS[i],
                POSITIONS[j]
            ) <= COMMUNICATION_RANGE:

                graph[i].append(j)

    base_neighbors = []

    for i in range(len(POSITIONS)):

        if distance(
            POSITIONS[i],
            BASE_STATION
        ) <= COMMUNICATION_RANGE:

            base_neighbors.append(i)

    return graph, base_neighbors


# ============================================================
# ROUTE SEARCH
# ============================================================

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

        relay_hops = len(path) - 1

        if relay_hops >= MAX_RELAY_HOPS:
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


# ============================================================
# CREATE MESH ONCE
# ============================================================

graph, base_neighbors = build_mesh()


# ============================================================
# RUN ONE ROBUSTNESS TEST
# ============================================================

def run_test(packet_success_rate):

    rng = np.random.default_rng(
        RNG_SEED
    )

    detected_alerts = 0
    delivered_alerts = 0
    transmission_failures = 0
    no_route_alerts = 0

    total_transmissions = 0

    for _ in range(N_ALERTS):

        # ----------------------------------------------------
        # Simulated ground truth
        # ----------------------------------------------------

        actual_survivor = (
            rng.random() < 0.50
        )

        # ----------------------------------------------------
        # Simulated sensor conditions
        # ----------------------------------------------------

        if actual_survivor:

            scenario = rng.choice([
                "buried",
                "surface",
                "silent"
            ])

            if scenario == "buried":

                rf = 0.80
                sound = 0.50
                pir = 0.40
                ir = 0.40

            elif scenario == "surface":

                rf = 1.00
                sound = 0.80
                pir = 1.00
                ir = 1.00

            else:

                rf = 1.00
                sound = 0.00
                pir = 0.40
                ir = 0.40

        else:

            scenario = rng.choice([
                "empty",
                "environmental_sound",
                "motion_only",
                "thermal_only"
            ])

            if scenario == "empty":

                rf = 0.00
                sound = 0.00
                pir = 0.00
                ir = 0.00

            elif scenario == "environmental_sound":

                rf = 0.00
                sound = 1.00
                pir = 0.00
                ir = 0.00

            elif scenario == "motion_only":

                rf = 0.00
                sound = 0.00
                pir = 1.00
                ir = 0.00

            else:

                rf = 0.00
                sound = 0.00
                pir = 0.00
                ir = 1.00

        # ----------------------------------------------------
        # Confidence fusion
        # ----------------------------------------------------

        confidence = (
            RF_WEIGHT * rf +
            SOUND_WEIGHT * sound +
            PIR_WEIGHT * pir +
            IR_WEIGHT * ir
        )

        detected = (
            confidence >=
            CONFIDENCE_THRESHOLD
        )

        if not detected:
            continue

        detected_alerts += 1

        # ----------------------------------------------------
        # Select beacon
        # ----------------------------------------------------

        source = rng.integers(
            0,
            len(POSITIONS)
        )

        route = find_route(
            source,
            graph,
            base_neighbors
        )

        if route is None:

            no_route_alerts += 1
            continue

        # ----------------------------------------------------
        # Number of wireless transmissions
        # ----------------------------------------------------

        relay_hops = len(route) - 1

        transmissions = relay_hops + 1

        total_transmissions += transmissions

        # ----------------------------------------------------
        # Packet transmission
        # ----------------------------------------------------

        delivered = True

        for _ in range(transmissions):

            if (
                rng.random()
                > packet_success_rate
            ):

                delivered = False
                break

        if delivered:

            delivered_alerts += 1

        else:

            transmission_failures += 1

    if detected_alerts > 0:

        delivery_rate = (
            delivered_alerts /
            detected_alerts *
            100
        )

    else:

        delivery_rate = 0

    if detected_alerts > 0:

        route_failure_rate = (
            no_route_alerts /
            detected_alerts *
            100
        )

    else:

        route_failure_rate = 0

    return {
        "packet_success": packet_success_rate,
        "detected": detected_alerts,
        "delivered": delivered_alerts,
        "transmission_failures":
            transmission_failures,
        "no_route": no_route_alerts,
        "delivery_rate": delivery_rate,
        "route_failure_rate":
            route_failure_rate,
        "average_transmissions": (
            total_transmissions /
            detected_alerts
            if detected_alerts > 0
            else 0
        )
    }


# ============================================================
# RUN ALL TESTS
# ============================================================

print("=" * 78)
print("SMART DISASTER RESCUE BEACON")
print("END-TO-END ROBUSTNESS SIMULATION")
print("=" * 78)

print()
print("Fixed parameters:")
print(f"Communication range : {COMMUNICATION_RANGE:.1f} m")
print(f"Maximum relay hops  : {MAX_RELAY_HOPS}")
print(f"Number of alerts    : {N_ALERTS}")
print()

results = []

for rate in PACKET_SUCCESS_RATES:

    result = run_test(rate)

    results.append(result)

    print(
        f"Packet success rate: "
        f"{rate:.2f}"
    )

    print(
        f"Detected alerts       : "
        f"{result['detected']}"
    )

    print(
        f"Delivered alerts      : "
        f"{result['delivered']}"
    )

    print(
        f"Transmission failures : "
        f"{result['transmission_failures']}"
    )

    print(
        f"No-route alerts       : "
        f"{result['no_route']}"
    )

    print(
        f"Alert delivery rate   : "
        f"{result['delivery_rate']:.2f}%"
    )

    print(
        f"Average transmissions : "
        f"{result['average_transmissions']:.2f}"
    )

    print("-" * 78)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(
    f"{'Packet':>10} | "
    f"{'Delivered':>10} | "
    f"{'Failed':>8} | "
    f"{'Delivery':>10}"
)

print("-" * 78)

for result in results:

    print(
        f"{result['packet_success']:>9.0%} | "
        f"{result['delivered']:>10} | "
        f"{result['transmission_failures']:>8} | "
        f"{result['delivery_rate']:>9.2f}%"
    )

print()
print("=" * 78)
print("ROBUSTNESS SIMULATION COMPLETE")
print("=" * 78)