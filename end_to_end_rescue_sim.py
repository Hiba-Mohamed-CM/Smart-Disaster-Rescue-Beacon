import numpy as np
from collections import deque

# ============================================================
# SMART DISASTER RESCUE BEACON
# END-TO-END SYSTEM SIMULATION
#
# Sensor fusion -> GPS tagging -> LoRa mesh -> Base station
# ============================================================

RNG_SEED = 42
N_ALERTS = 1000

# Confidence fusion weights
RF_WEIGHT = 0.50
SOUND_WEIGHT = 0.20
PIR_WEIGHT = 0.15
IR_WEIGHT = 0.15

CONFIDENCE_THRESHOLD = 0.50

# Mesh parameters
COMMUNICATION_RANGE = 3.0
MAX_RELAY_HOPS = 6
PACKET_SUCCESS_RATE = 0.95

BASE_STATION = np.array([0.0, -2.0])

# Same beacon topology used previously
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
# SENSOR CONFIDENCE
# ============================================================

def calculate_confidence(rf, sound, pir, ir):

    confidence = (
        RF_WEIGHT * rf +
        SOUND_WEIGHT * sound +
        PIR_WEIGHT * pir +
        IR_WEIGHT * ir
    )

    return confidence


def survivor_decision(confidence):

    return confidence >= CONFIDENCE_THRESHOLD


# ============================================================
# BUILD LORA MESH
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
# FIND ROUTE TO BASE
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

        # Current beacon can transmit directly to base
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
# SIMULATION
# ============================================================

rng = np.random.default_rng(RNG_SEED)

graph, base_neighbors = build_mesh()

survivor_cases = 0
false_alarm_cases = 0
missed_cases = 0
correct_rejections = 0

delivered_alerts = 0
undelivered_alerts = 0
no_route_alerts = 0

routes_used = []
confidence_values = []


for alert_number in range(N_ALERTS):

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    actual_survivor = (
        rng.random() < 0.50
    )

    # --------------------------------------------------------
    # Simulated sensor confidence
    #
    # These are system-level assumptions, NOT measured
    # sensor accuracies.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Sensor fusion
    # --------------------------------------------------------

    confidence = calculate_confidence(
        rf,
        sound,
        pir,
        ir
    )

    detected = survivor_decision(
        confidence
    )

    confidence_values.append(
        confidence
    )

    # --------------------------------------------------------
    # Detection statistics
    # --------------------------------------------------------

    if actual_survivor and detected:

        survivor_cases += 1

    elif actual_survivor and not detected:

        missed_cases += 1

    elif not actual_survivor and detected:

        false_alarm_cases += 1

    else:

        correct_rejections += 1

    # --------------------------------------------------------
    # Only detected survivors generate an alert
    # --------------------------------------------------------

    if not detected:
        continue

    # GPS location of the beacon
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

    # --------------------------------------------------------
    # Simulate packet transmission
    # --------------------------------------------------------

    relay_hops = len(route) - 1

    # relay transmissions + final beacon-to-base transmission
    total_transmissions = relay_hops + 1

    delivered = True

    for _ in range(total_transmissions):

        if rng.random() > PACKET_SUCCESS_RATE:

            delivered = False
            break

    if delivered:

        delivered_alerts += 1
        routes_used.append(
            total_transmissions
        )

    else:

        undelivered_alerts += 1


# ============================================================
# CALCULATE METRICS
# ============================================================

detection_accuracy = (
    (survivor_cases + correct_rejections)
    / N_ALERTS * 100
)

detection_recall = (
    survivor_cases
    / (survivor_cases + missed_cases)
    * 100
)

false_alarm_rate = (
    false_alarm_cases
    / (
        false_alarm_cases +
        correct_rejections
    )
    * 100
)

detected_alerts = (
    survivor_cases +
    false_alarm_cases
)

if detected_alerts > 0:

    alert_delivery_rate = (
        delivered_alerts /
        detected_alerts *
        100
    )

else:

    alert_delivery_rate = 0


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("SMART DISASTER RESCUE BEACON")
print("END-TO-END SYSTEM SIMULATION")
print("=" * 70)

print()

print("SYSTEM PARAMETERS")
print("-" * 70)

print(
    f"Communication range       : "
    f"{COMMUNICATION_RANGE:.1f} m"
)

print(
    f"Maximum relay hops        : "
    f"{MAX_RELAY_HOPS}"
)

print(
    f"Packet success probability: "
    f"{PACKET_SUCCESS_RATE:.2f}"
)

print(
    f"Confidence threshold      : "
    f"{CONFIDENCE_THRESHOLD:.2f}"
)

print()

print("SENSOR FUSION")
print("-" * 70)

print(
    f"RF weight                 : "
    f"{RF_WEIGHT:.2f}"
)

print(
    f"Sound weight              : "
    f"{SOUND_WEIGHT:.2f}"
)

print(
    f"PIR weight                : "
    f"{PIR_WEIGHT:.2f}"
)

print(
    f"IR weight                 : "
    f"{IR_WEIGHT:.2f}"
)

print()

print("DETECTION RESULTS")
print("-" * 70)

print(
    f"Correct survivor detections: "
    f"{survivor_cases}"
)

print(
    f"Missed survivors           : "
    f"{missed_cases}"
)

print(
    f"False alarms               : "
    f"{false_alarm_cases}"
)

print(
    f"Correct rejections         : "
    f"{correct_rejections}"
)

print(
    f"Detection accuracy         : "
    f"{detection_accuracy:.2f}%"
)

print(
    f"Survivor recall            : "
    f"{detection_recall:.2f}%"
)

print(
    f"False alarm rate           : "
    f"{false_alarm_rate:.2f}%"
)

print()

print("MESH DELIVERY")
print("-" * 70)

print(
    f"Detected alerts            : "
    f"{detected_alerts}"
)

print(
    f"Delivered to base station  : "
    f"{delivered_alerts}"
)

print(
    f"Transmission failures      : "
    f"{undelivered_alerts}"
)

print(
    f"No-route alerts            : "
    f"{no_route_alerts}"
)

print(
    f"Alert delivery rate        : "
    f"{alert_delivery_rate:.2f}%"
)

if routes_used:

    print(
        f"Average transmissions     : "
        f"{np.mean(routes_used):.2f}"
    )

else:

    print(
        "Average transmissions     : 0.00"
    )

print()

print("=" * 70)
print("END-TO-END SIMULATION COMPLETE")
print("=" * 70)