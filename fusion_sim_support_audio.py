import numpy as np

# ============================================================
# RF + SUPPORTING AUDIO FUSION SIMULATION
# Correct handling of distance information
# ============================================================

RF_FILE = "rf_pool.npz"
SOUND_FILE = "sound_pool.npz"

RNG = np.random.default_rng(42)
N_TRIALS = 5000


# ------------------------------------------------------------
# Load pools
# ------------------------------------------------------------

rf = np.load(RF_FILE)
sound = np.load(SOUND_FILE)

print("Loaded pools:")

for k in rf.files:
    print(f"  RF {k:12s}: {np.asarray(rf[k]).shape}")

for k in sound.files:
    print(f"  Sound {k:8s}: {np.asarray(sound[k]).shape}")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def sample_rows(x, n):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), size=n)
    return x[idx]


def rf_score(x):
    """
    RF trials contain 5 decision windows.
    Use the 75th percentile of the five windows.
    """
    x = np.asarray(x)

    if x.ndim == 2:
        return np.percentile(x, 75, axis=1)

    return x


def audio_score(x):
    """
    Audio trials contain 5 one-second windows.
    Use the strongest window.
    """
    x = np.asarray(x)

    if x.ndim == 2:
        return np.max(x, axis=1)

    return x


def percentile_tail_score(values, background):
    """
    Convert a value into its percentile relative to
    the empty/noise background.
    """
    background = np.sort(np.asarray(background))

    idx = np.searchsorted(
        background,
        values,
        side="right"
    )

    return idx / len(background)


def metrics(y_true, y_pred):

    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    accuracy = (tp + tn) / len(y_true)

    precision = (
        tp / (tp + fp)
        if tp + fp else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall else 0
    )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "FPR": fpr,
        "F1": f1
    }


def print_metrics(title, m):

    print()
    print(title)
    print("-" * len(title))

    print(
        f"TP={m['TP']} "
        f"TN={m['TN']} "
        f"FP={m['FP']} "
        f"FN={m['FN']}"
    )

    print(f"Accuracy  {m['Accuracy'] * 100:.2f}%")
    print(f"Precision {m['Precision'] * 100:.2f}%")
    print(f"Recall    {m['Recall'] * 100:.2f}%")
    print(f"FPR       {m['FPR'] * 100:.2f}%")
    print(f"F1        {m['F1'] * 100:.2f}%")


# ------------------------------------------------------------
# Background
# ------------------------------------------------------------

rf_empty = rf_score(rf["empty"])
sound_noise = audio_score(sound["noise"])

print()
print("Background distributions:")
print(f"  RF empty samples   : {len(rf_empty)}")
print(f"  Sound noise samples: {len(sound_noise)}")


# ------------------------------------------------------------
# RF human pool
#
# IMPORTANT:
# dist_stand and dist_lying are distance information,
# NOT RF signal measurements.
#
# Therefore they are NOT included in human_pool.
# ------------------------------------------------------------

stand = np.asarray(rf["stand"])
lying = np.asarray(rf["lying"])

human_pool = np.concatenate(
    [
        stand,
        lying
    ],
    axis=0
)

print()
print(f"Human RF measurement pool: {human_pool.shape}")


# ------------------------------------------------------------
# Simulation scenarios
#
# 1. empty + noise
# 2. human + noise
# 3. empty + call
# 4. human + call
# ------------------------------------------------------------

n_each = N_TRIALS // 4

scenarios = [

    {
        "name": "empty + noise",
        "rf": sample_rows(
            rf["empty"],
            n_each
        ),
        "sound": sample_rows(
            sound["noise"],
            n_each
        ),
        "truth": np.zeros(
            n_each,
            dtype=int
        )
    },

    {
        "name": "human + noise",
        "rf": sample_rows(
            human_pool,
            n_each
        ),
        "sound": sample_rows(
            sound["noise"],
            n_each
        ),
        "truth": np.ones(
            n_each,
            dtype=int
        )
    },

    {
        "name": "empty + call",
        "rf": sample_rows(
            rf["empty"],
            n_each
        ),
        "sound": sample_rows(
            sound["buried_calling"],
            n_each
        ),
        "truth": np.zeros(
            n_each,
            dtype=int
        )
    },

    {
        "name": "human + call",
        "rf": sample_rows(
            human_pool,
            n_each
        ),
        "sound": sample_rows(
            sound["buried_calling"],
            n_each
        ),
        "truth": np.ones(
            n_each,
            dtype=int
        )
    }
]


# ------------------------------------------------------------
# Calculate evidence scores
# ------------------------------------------------------------

all_rf_scores = []
all_audio_scores = []
all_truth = []
all_names = []

for sc in scenarios:

    rf_values = rf_score(sc["rf"])
    audio_values = audio_score(sc["sound"])

    rf_evidence = percentile_tail_score(
        rf_values,
        rf_empty
    )

    audio_evidence = percentile_tail_score(
        audio_values,
        sound_noise
    )

    all_rf_scores.append(rf_evidence)
    all_audio_scores.append(audio_evidence)
    all_truth.append(sc["truth"])

    all_names.extend(
        [sc["name"]] * len(rf_evidence)
    )


rf_scores = np.concatenate(all_rf_scores)
audio_scores = np.concatenate(all_audio_scores)
truth = np.concatenate(all_truth)
all_names = np.asarray(all_names)


# ------------------------------------------------------------
# Detection thresholds
# ------------------------------------------------------------

RF_THRESHOLD = 0.95

AUDIO_SUPPORT_THRESHOLD = 0.90

RF_SUPPORT_MARGIN = 0.10

print()
print("Fusion settings:")
print(f"RF threshold            : {RF_THRESHOLD:.3f}")
print(
    f"Audio support threshold : "
    f"{AUDIO_SUPPORT_THRESHOLD:.3f}"
)
print(
    f"RF support margin       : "
    f"{RF_SUPPORT_MARGIN:.3f}"
)


# ------------------------------------------------------------
# RF ONLY
# ------------------------------------------------------------

rf_prediction = (
    rf_scores >= RF_THRESHOLD
)


# ------------------------------------------------------------
# SUPPORTING AUDIO
#
# Audio cannot independently declare a human.
#
# It can only help if RF is already close to
# its detection threshold.
# ------------------------------------------------------------

audio_support = (
    audio_scores >= AUDIO_SUPPORT_THRESHOLD
)

rf_near_threshold = (
    rf_scores >=
    (RF_THRESHOLD - RF_SUPPORT_MARGIN)
)

fusion_prediction = (

    rf_scores >= RF_THRESHOLD

) | (

    rf_near_threshold &
    audio_support
)


# ------------------------------------------------------------
# Overall results
# ------------------------------------------------------------

print()
print("=" * 60)
print("SIMULATION RESULTS")
print("=" * 60)

rf_result = metrics(
    truth,
    rf_prediction
)

fusion_result = metrics(
    truth,
    fusion_prediction
)

print_metrics(
    "RF ONLY",
    rf_result
)

print_metrics(
    "RF + SUPPORTING AUDIO",
    fusion_result
)


# ------------------------------------------------------------
# Scenario results
# ------------------------------------------------------------

print()
print("=" * 60)
print("DETECTION RATE BY SCENARIO")
print("=" * 60)

for name in [
    "empty + noise",
    "human + noise",
    "empty + call",
    "human + call"
]:

    mask = all_names == name

    rf_rate = np.mean(
        rf_prediction[mask]
    )

    audio_rate = np.mean(
        audio_support[mask]
    )

    fusion_rate = np.mean(
        fusion_prediction[mask]
    )

    print(
        f"{name:18s} | "
        f"RF {rf_rate * 100:5.1f}% | "
        f"Audio support {audio_rate * 100:5.1f}% | "
        f"Fusion {fusion_rate * 100:5.1f}%"
    )


# ------------------------------------------------------------
# DISTANCE ANALYSIS
#
# dist_stand and dist_lying tell us the distance
# associated with the corresponding RF trials.
#
# They are NOT used as RF signal values.
# ------------------------------------------------------------

print()
print("=" * 60)
print("DISTANCE ANALYSIS")
print("=" * 60)


def analyze_distance(
    rf_measurements,
    distances,
    name
):

    rf_measurements = np.asarray(
        rf_measurements
    )

    distances = np.asarray(
        distances
    )

    # Each RF measurement row corresponds to
    # one distance entry.

    unique_distances = np.unique(
        distances
    )

    print()
    print(name)

    for distance in unique_distances:

        mask = (
            distances == distance
        )

        measurements = (
            rf_measurements[mask]
        )

        scores = rf_score(
            measurements
        )

        evidence = percentile_tail_score(
            scores,
            rf_empty
        )

        prediction = (
            evidence >= RF_THRESHOLD
        )

        print(
            f"  Distance {distance:.1f} m | "
            f"Trials {np.sum(mask):4d} | "
            f"Detection "
            f"{np.mean(prediction) * 100:5.1f}% | "
            f"Median RF evidence "
            f"{np.median(evidence):.3f}"
        )


analyze_distance(
    rf["stand"],
    rf["dist_stand"],
    "STANDING PERSON"
)

analyze_distance(
    rf["lying"],
    rf["dist_lying"],
    "LYING PERSON"
)


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

np.savez(
    "fusion_results_support_audio.npz",

    truth=truth,

    rf_scores=rf_scores,

    audio_scores=audio_scores,

    rf_prediction=rf_prediction,

    fusion_prediction=fusion_prediction
)


print()
print("Saved:")
print("  fusion_results_support_audio.npz")