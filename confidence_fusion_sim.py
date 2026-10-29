# Smart Disaster Rescue Beacon
# System-level confidence fusion simulation

# Sensor weights
RF_WEIGHT = 0.50
SOUND_WEIGHT = 0.20
PIR_WEIGHT = 0.15
IR_WEIGHT = 0.15

CONFIDENCE_THRESHOLD = 0.50


def calculate_confidence(rf, sound, pir, ir):
    confidence = (
        RF_WEIGHT * rf +
        SOUND_WEIGHT * sound +
        PIR_WEIGHT * pir +
        IR_WEIGHT * ir
    )

    return confidence


scenarios = [
    {
        "name": "Empty environment",
        "rf": 0.0,
        "sound": 0.0,
        "pir": 0.0,
        "ir": 0.0,
        "expected": "NO SURVIVOR"
    },

    {
        "name": "Buried victim - RF + sound",
        "rf": 1.0,
        "sound": 1.0,
        "pir": 0.0,
        "ir": 0.0,
        "expected": "SURVIVOR"
    },

    {
        "name": "Buried silent victim - RF only",
        "rf": 1.0,
        "sound": 0.0,
        "pir": 0.0,
        "ir": 0.0,
        "expected": "SURVIVOR"
    },

    {
        "name": "Surface victim",
        "rf": 1.0,
        "sound": 1.0,
        "pir": 1.0,
        "ir": 1.0,
        "expected": "SURVIVOR"
    },

    {
        "name": "Surface silent victim",
        "rf": 1.0,
        "sound": 0.0,
        "pir": 1.0,
        "ir": 1.0,
        "expected": "SURVIVOR"
    },

    {
        "name": "Environmental sound only",
        "rf": 0.0,
        "sound": 1.0,
        "pir": 0.0,
        "ir": 0.0,
        "expected": "NO SURVIVOR"
    },

    {
        "name": "Motion only",
        "rf": 0.0,
        "sound": 0.0,
        "pir": 1.0,
        "ir": 0.0,
        "expected": "NO SURVIVOR"
    },

    {
        "name": "Thermal indication only",
        "rf": 0.0,
        "sound": 0.0,
        "pir": 0.0,
        "ir": 1.0,
        "expected": "NO SURVIVOR"
    }
]


print("=" * 75)
print("SMART DISASTER RESCUE BEACON")
print("CONFIDENCE-BASED SENSOR FUSION SIMULATION")
print("=" * 75)

print()
print("Sensor weights:")
print(f"RF     : {RF_WEIGHT:.2f}")
print(f"Sound  : {SOUND_WEIGHT:.2f}")
print(f"PIR    : {PIR_WEIGHT:.2f}")
print(f"IR     : {IR_WEIGHT:.2f}")
print(f"Threshold: {CONFIDENCE_THRESHOLD:.2f}")

print()
print("=" * 75)

for scenario in scenarios:

    confidence = calculate_confidence(
        scenario["rf"],
        scenario["sound"],
        scenario["pir"],
        scenario["ir"]
    )

    if confidence >= CONFIDENCE_THRESHOLD:
        decision = "SURVIVOR DETECTED"
    else:
        decision = "NO SURVIVOR"

    print()
    print(scenario["name"])
    print("-" * 75)

    print(
        f"RF={scenario['rf']:.1f} | "
        f"Sound={scenario['sound']:.1f} | "
        f"PIR={scenario['pir']:.1f} | "
        f"IR={scenario['ir']:.1f}"
    )

    print(f"Confidence score : {confidence:.2f}")
    print(f"Decision         : {decision}")
    print(f"Expected         : {scenario['expected']}")


print()
print("=" * 75)
print("SIMULATION COMPLETE")
print("=" * 75)