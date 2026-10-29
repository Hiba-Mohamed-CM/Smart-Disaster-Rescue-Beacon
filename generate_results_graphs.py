import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = "results_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# GRAPH 1: RF ONLY vs RF + SUPPORTING AUDIO
# ============================================================

methods = ["RF only", "RF + supporting audio"]

accuracy = [91.08, 91.32]
precision = [94.93, 94.15]
recall = [86.80, 88.12]
f1 = [90.68, 91.03]

x = np.arange(len(methods))
width = 0.18

plt.figure(figsize=(9, 6))

plt.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5 * width, precision, width, label="Precision")
plt.bar(x + 0.5 * width, recall, width, label="Recall")
plt.bar(x + 1.5 * width, f1, width, label="F1-score")

plt.xticks(x, methods)
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)
plt.title("RF-only vs RF + Supporting Audio")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "rf_vs_audio.png"),
    dpi=300
)
plt.show()


# ============================================================
# GRAPH 2: COMMUNICATION RANGE vs ALERT DELIVERY
# ============================================================

communication_range = [1.5, 2.0, 2.5, 3.0, 3.5]
alert_delivery = [0.0, 62.7, 62.7, 84.2, 84.2]

plt.figure(figsize=(9, 6))

plt.plot(
    communication_range,
    alert_delivery,
    marker="o",
    linewidth=2
)

plt.xlabel("Communication range (m)")
plt.ylabel("Alert delivery rate (%)")
plt.title("Communication Range vs Alert Delivery")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "range_vs_delivery.png"),
    dpi=300
)
plt.show()


# ============================================================
# GRAPH 3: PACKET SUCCESS RATE vs ALERT DELIVERY
# ============================================================

packet_success = [80, 85, 90, 95, 98, 100]
delivery_rate = [45.60, 56.87, 69.59, 85.71, 93.39, 100.00]

plt.figure(figsize=(9, 6))

plt.plot(
    packet_success,
    delivery_rate,
    marker="o",
    linewidth=2
)

plt.xlabel("Per-transmission packet success rate (%)")
plt.ylabel("End-to-end alert delivery (%)")
plt.title("Packet Success Rate vs End-to-End Alert Delivery")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "packet_success_vs_delivery.png"),
    dpi=300
)
plt.show()


# ============================================================
# GRAPH 4: END-TO-END SYSTEM RESULTS
# ============================================================

categories = [
    "Detected alerts",
    "Delivered to base",
    "Transmission failures",
    "No-route"
]

values = [518, 444, 74, 0]

plt.figure(figsize=(9, 6))

bars = plt.bar(categories, values)

plt.ylabel("Number of alerts")
plt.title("End-to-End Rescue Simulation Results")
plt.grid(axis="y", alpha=0.3)

# Display values above bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 8,
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "end_to_end_results.png"),
    dpi=300
)
plt.show()


print("\n========================================")
print("ALL GRAPHS GENERATED SUCCESSFULLY")
print("========================================")
print(f"Saved inside: {os.path.abspath(OUTPUT_DIR)}")
print("\nFiles:")
print("1. rf_vs_audio.png")
print("2. range_vs_delivery.png")
print("3. packet_success_vs_delivery.png")
print("4. end_to_end_results.png")