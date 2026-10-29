import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FINAL RESULTS ANALYSIS
# RF + SUPPORTING AUDIO
# ============================================================

# Load previously generated simulation results
results = np.load("fusion_results_support_audio.npz")

truth = results["truth"]
rf_scores = results["rf_scores"]
audio_scores = results["audio_scores"]
rf_prediction = results["rf_prediction"]
fusion_prediction = results["fusion_prediction"]


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    TP = np.sum(y_true & y_pred)
    TN = np.sum(~y_true & ~y_pred)
    FP = np.sum(~y_true & y_pred)
    FN = np.sum(y_true & ~y_pred)

    accuracy = (TP + TN) / len(y_true)

    precision = (
        TP / (TP + FP)
        if TP + FP > 0 else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0 else 0
    )

    fpr = (
        FP / (FP + TN)
        if FP + TN > 0 else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "F1": f1
    }


rf = calculate_metrics(
    truth,
    rf_prediction
)

fusion = calculate_metrics(
    truth,
    fusion_prediction
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("=" * 65)
print("FINAL PERFORMANCE RESULTS")
print("=" * 65)

print()
print(
    f"{'Metric':<15}"
    f"{'RF Only':>15}"
    f"{'RF + Audio':>18}"
)

print("-" * 48)

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "FPR",
    "F1"
]:

    print(
        f"{metric:<15}"
        f"{rf[metric] * 100:>14.2f}%"
        f"{fusion[metric] * 100:>17.2f}%"
    )


# ------------------------------------------------------------
# Confusion matrices
# ------------------------------------------------------------

def confusion_matrix_values(y_true, y_pred):

    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    TP = np.sum(y_true & y_pred)
    TN = np.sum(~y_true & ~y_pred)
    FP = np.sum(~y_true & y_pred)
    FN = np.sum(y_true & ~y_pred)

    return np.array([
        [TN, FP],
        [FN, TP]
    ])


rf_cm = confusion_matrix_values(
    truth,
    rf_prediction
)

fusion_cm = confusion_matrix_values(
    truth,
    fusion_prediction
)


# ------------------------------------------------------------
# Plot confusion matrices
# ------------------------------------------------------------

def plot_confusion_matrix(
    matrix,
    title,
    filename
):

    plt.figure(figsize=(5, 4))

    plt.imshow(matrix)

    plt.title(title)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["No Human", "Human"]
    )

    plt.yticks(
        [0, 1],
        ["No Human", "Human"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()


plot_confusion_matrix(
    rf_cm,
    "RF Only Confusion Matrix",
    "confusion_matrix_rf.png"
)

plot_confusion_matrix(
    fusion_cm,
    "RF + Audio Confusion Matrix",
    "confusion_matrix_fusion.png"
)


# ------------------------------------------------------------
# Performance comparison graph
# ------------------------------------------------------------

metrics_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

rf_values = [
    rf[x] * 100
    for x in metrics_names
]

fusion_values = [
    fusion[x] * 100
    for x in metrics_names
]

x = np.arange(
    len(metrics_names)
)

width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    rf_values,
    width,
    label="RF Only"
)

plt.bar(
    x + width / 2,
    fusion_values,
    width,
    label="RF + Audio"
)

plt.xticks(
    x,
    metrics_names
)

plt.ylabel("Percentage (%)")
plt.title("Performance Comparison")

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "performance_comparison.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Save numerical results
# ------------------------------------------------------------

with open(
    "final_results.txt",
    "w"
) as f:

    f.write(
        "FINAL PERFORMANCE RESULTS\n"
    )

    f.write(
        "=========================\n\n"
    )

    for name, data in [
        ("RF ONLY", rf),
        ("RF + SUPPORTING AUDIO", fusion)
    ]:

        f.write(
            name + "\n"
        )

        f.write(
            "-------------------------\n"
        )

        for metric in [
            "TP",
            "TN",
            "FP",
            "FN"
        ]:

            f.write(
                f"{metric}: "
                f"{data[metric]}\n"
            )

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "FPR",
            "F1"
        ]:

            f.write(
                f"{metric}: "
                f"{data[metric] * 100:.2f}%\n"
            )

        f.write("\n")


print()
print("=" * 65)
print("FILES GENERATED")
print("=" * 65)

print("performance_comparison.png")
print("confusion_matrix_rf.png")
print("confusion_matrix_fusion.png")
print("final_results.txt")

print()
print("Final results analysis completed.")