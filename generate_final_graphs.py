import matplotlib.pyplot as plt

# ============================================================
# FINAL RESULTS
# ============================================================

# ---------------- DISTANCE DATA ----------------

distances = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

standing_detection = [
    100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 100.0
]

lying_detection = [
    100.0, 77.1, 83.0, 89.0,
    69.4, 34.2, 40.4
]

# ============================================================
# GRAPH 1: DETECTION RATE VS DISTANCE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    distances,
    standing_detection,
    marker="o",
    linewidth=2,
    label="Standing Person"
)

plt.plot(
    distances,
    lying_detection,
    marker="o",
    linewidth=2,
    label="Lying Person"
)

plt.xlabel("Distance (m)")
plt.ylabel("Detection Rate (%)")
plt.title("RF Detection Rate vs Distance")

plt.xticks(distances)
plt.yticks(range(0, 101, 10))
plt.ylim(0, 105)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "final_distance_detection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 2: FALSE POSITIVE RATE
# ============================================================

methods = [
    "RF Only",
    "RF + Supporting Audio"
]

fpr = [
    4.64,
    5.48
]

plt.figure(figsize=(8, 6))

bars = plt.bar(methods, fpr)

plt.ylabel("False Positive Rate (%)")
plt.title("False Positive Rate Comparison")

plt.ylim(0, 7)

# Add values above bars
for bar, value in zip(bars, fpr):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "final_fpr_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nGraphs generated successfully!")
print("--------------------------------")
print("1. final_distance_detection.png")
print("2. final_fpr_comparison.png")