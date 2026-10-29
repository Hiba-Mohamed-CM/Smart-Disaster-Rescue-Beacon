import numpy as np

RF_FILE = "rf_pool.npz"
SOUND_FILE = "sound_pool.npz"

N_TRIALS = 5000
SEEDS = [1, 2, 3, 4, 5]

RF_THRESHOLD = 0.95
AUDIO_SUPPORT_THRESHOLD = 0.90
RF_SUPPORT_MARGIN = 0.10


def run_simulation(seed):

    RNG = np.random.default_rng(seed)

    rf = np.load(RF_FILE)
    sound = np.load(SOUND_FILE)

    def sample_rows(x, n):
        x = np.asarray(x)
        idx = RNG.integers(0, len(x), size=n)
        return x[idx]

    def rf_score(x):
        x = np.asarray(x)

        if x.ndim == 2:
            return np.percentile(x, 75, axis=1)

        return x

    def audio_score(x):
        x = np.asarray(x)

        if x.ndim == 2:
            return np.max(x, axis=1)

        return x

    def percentile_tail_score(values, background):
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

    # ------------------------------------------------------------
    # Background
    # ------------------------------------------------------------

    rf_empty = rf_score(rf["empty"])
    sound_noise = audio_score(sound["noise"])

    # ------------------------------------------------------------
    # Human RF pool
    # ------------------------------------------------------------

    stand = np.asarray(rf["stand"])
    lying = np.asarray(rf["lying"])

    human_pool = np.concatenate(
        [stand, lying],
        axis=0
    )

    # ------------------------------------------------------------
    # Four scenarios
    # ------------------------------------------------------------

    n_each = N_TRIALS // 4

    scenarios = [

        {
            "name": "empty + noise",
            "rf": sample_rows(rf["empty"], n_each),
            "sound": sample_rows(sound["noise"], n_each),
            "truth": np.zeros(n_each, dtype=int)
        },

        {
            "name": "human + noise",
            "rf": sample_rows(human_pool, n_each),
            "sound": sample_rows(sound["noise"], n_each),
            "truth": np.ones(n_each, dtype=int)
        },

        {
            "name": "empty + call",
            "rf": sample_rows(rf["empty"], n_each),
            "sound": sample_rows(sound["buried_calling"], n_each),
            "truth": np.zeros(n_each, dtype=int)
        },

        {
            "name": "human + call",
            "rf": sample_rows(human_pool, n_each),
            "sound": sample_rows(sound["buried_calling"], n_each),
            "truth": np.ones(n_each, dtype=int)
        }
    ]

    all_rf_scores = []
    all_audio_scores = []
    all_truth = []

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

    rf_scores = np.concatenate(all_rf_scores)
    audio_scores = np.concatenate(all_audio_scores)
    truth = np.concatenate(all_truth)

    # ------------------------------------------------------------
    # RF ONLY
    # ------------------------------------------------------------

    rf_prediction = (
        rf_scores >= RF_THRESHOLD
    )

    # ------------------------------------------------------------
    # SUPPORTING AUDIO
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
    # Results
    # ------------------------------------------------------------

    rf_result = metrics(
        truth,
        rf_prediction
    )

    fusion_result = metrics(
        truth,
        fusion_prediction
    )

    return rf_result, fusion_result


# ============================================================
# ROBUSTNESS TEST
# ============================================================

print("=" * 78)
print("RF + SUPPORTING AUDIO ROBUSTNESS TEST")
print("=" * 78)

print()
print("Seeds:", SEEDS)
print("Trials per seed:", N_TRIALS)
print()

rf_results = []
fusion_results = []

for seed in SEEDS:

    rf_result, fusion_result = run_simulation(seed)

    rf_results.append(rf_result)
    fusion_results.append(fusion_result)

    print(f"SEED {seed}")
    print("-" * 78)

    print(
        f"RF Only       : "
        f"Accuracy {rf_result['Accuracy']*100:.2f}% | "
        f"Precision {rf_result['Precision']*100:.2f}% | "
        f"Recall {rf_result['Recall']*100:.2f}% | "
        f"F1 {rf_result['F1']*100:.2f}% | "
        f"FPR {rf_result['FPR']*100:.2f}%"
    )

    print(
        f"RF + Audio    : "
        f"Accuracy {fusion_result['Accuracy']*100:.2f}% | "
        f"Precision {fusion_result['Precision']*100:.2f}% | "
        f"Recall {fusion_result['Recall']*100:.2f}% | "
        f"F1 {fusion_result['F1']*100:.2f}% | "
        f"FPR {fusion_result['FPR']*100:.2f}%"
    )

    print()


# ============================================================
# MEAN AND STANDARD DEVIATION
# ============================================================

def summarize(results):

    metrics_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "FPR",
        "F1"
    ]

    summary = {}

    for name in metrics_names:

        values = np.array([
            result[name]
            for result in results
        ])

        summary[name] = (
            np.mean(values),
            np.std(values, ddof=1)
        )

    return summary


rf_summary = summarize(rf_results)
fusion_summary = summarize(fusion_results)


print("=" * 78)
print("ROBUSTNESS SUMMARY")
print("=" * 78)

print()
print("RF ONLY")
print("-" * 78)

for name, (mean, std) in rf_summary.items():

    print(
        f"{name:10s}: "
        f"{mean*100:.2f}% ± {std*100:.2f}%"
    )


print()
print("RF + SUPPORTING AUDIO")
print("-" * 78)

for name, (mean, std) in fusion_summary.items():

    print(
        f"{name:10s}: "
        f"{mean*100:.2f}% ± {std*100:.2f}%"
    )


print()
print("=" * 78)
print("ROBUSTNESS TEST COMPLETE")
print("=" * 78)