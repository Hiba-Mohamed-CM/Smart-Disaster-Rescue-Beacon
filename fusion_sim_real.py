"""
fusion_sim_real.py

A standalone fusion simulator for the RF/radar and audio pools produced by:
    build_rf_pool.py
    build_sound_pool.py

IMPORTANT:
This is a NEW fusion simulator written for the generated .npz files.
It is not the missing original fusion_sim_real.py from the source project.

Expected files in the same folder:
    rf_pool.npz
    sound_pool.npz

RF pool keys:
    empty, stand, lying
    dist_stand, dist_lying

Sound pool keys:
    noise, surface, buried_calling
    snr_surface, snr_buried

Run:
    python fusion_sim_real.py

Optional:
    python fusion_sim_real.py --trials 5000 --rf-weight 0.6 --sound-weight 0.4
"""

import argparse
import numpy as np


def load_pools(rf_path="rf_pool.npz", sound_path="sound_pool.npz"):
    try:
        rf = np.load(rf_path)
    except FileNotFoundError:
        raise SystemExit(
            f"Could not find {rf_path}. Put it in the same folder as this script."
        )

    try:
        sound = np.load(sound_path)
    except FileNotFoundError:
        raise SystemExit(
            f"Could not find {sound_path}. Put it in the same folder as this script."
        )

    required_rf = {"empty", "stand", "lying"}
    required_sound = {"noise", "surface", "buried_calling"}

    missing_rf = required_rf - set(rf.files)
    missing_sound = required_sound - set(sound.files)

    if missing_rf:
        raise SystemExit(f"RF pool is missing keys: {sorted(missing_rf)}")
    if missing_sound:
        raise SystemExit(f"Sound pool is missing keys: {sorted(missing_sound)}")

    return rf, sound


def trial_value(x):
    """Reduce a 5-window trial to one robust value."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return float(x)
    return float(np.median(x))


def median_mad(x):
    """Return median and MAD, with a safe fallback."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-9:
        mad = np.std(x)
    if mad < 1e-9:
        mad = 1.0
    return med, mad


def robust_score(x, med, mad):
    """Convert a measurement into a roughly standardized evidence score."""
    return (x - med) / (1.4826 * mad)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
    }


def print_metrics(name, m):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"TP={m['TP']:5d}  TN={m['TN']:5d}  FP={m['FP']:5d}  FN={m['FN']:5d}")
    print(f"Accuracy : {m['accuracy'] * 100:6.2f}%")
    print(f"Precision: {m['precision'] * 100:6.2f}%")
    print(f"Recall   : {m['recall'] * 100:6.2f}%")
    print(f"FPR      : {m['fpr'] * 100:6.2f}%")
    print(f"F1       : {m['f1'] * 100:6.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Monte-Carlo fusion simulation using the generated RF and sound pools."
    )
    parser.add_argument("--rf", default="rf_pool.npz")
    parser.add_argument("--sound", default="sound_pool.npz")
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--rf-weight", type=float, default=0.6)
    parser.add_argument("--sound-weight", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if args.rf_weight < 0 or args.sound_weight < 0:
        raise SystemExit("Fusion weights cannot be negative.")
    if args.rf_weight + args.sound_weight == 0:
        raise SystemExit("At least one fusion weight must be greater than zero.")

    rf_w = args.rf_weight / (args.rf_weight + args.sound_weight)
    sound_w = args.sound_weight / (args.rf_weight + args.sound_weight)

    rng = np.random.default_rng(args.seed)
    rf, sound = load_pools(args.rf, args.sound)

    # Convert each 5-window trial to one robust scalar.
    rf_empty = np.array([trial_value(v) for v in rf["empty"]])
    rf_stand = np.array([trial_value(v) for v in rf["stand"]])
    rf_lying = np.array([trial_value(v) for v in rf["lying"]])

    sound_noise = np.array([trial_value(v) for v in sound["noise"]])
    sound_surface = np.array([trial_value(v) for v in sound["surface"]])
    sound_buried = np.array([trial_value(v) for v in sound["buried_calling"]])

    if min(
        len(rf_empty),
        len(rf_stand),
        len(rf_lying),
        len(sound_noise),
        len(sound_surface),
        len(sound_buried),
    ) == 0:
        raise SystemExit("One of the pools contains zero trials.")

    # Baselines are learned only from the negative/background pools.
    rf_med, rf_mad = median_mad(rf_empty)
    sound_med, sound_mad = median_mad(sound_noise)

    # A 95th-percentile background threshold gives a simple false-alarm control.
    rf_threshold = np.percentile(rf_empty, 95)
    sound_threshold = np.percentile(sound_noise, 95)

    print("Loaded pools:")
    print(f"  RF empty : {len(rf_empty)} trials")
    print(f"  RF stand : {len(rf_stand)} trials")
    print(f"  RF lying : {len(rf_lying)} trials")
    print(f"  Sound noise     : {len(sound_noise)} trials")
    print(f"  Sound surface   : {len(sound_surface)} trials")
    print(f"  Sound buried    : {len(sound_buried)} trials")

    print("\nBackground thresholds (95th percentile):")
    print(f"  RF threshold    : {rf_threshold:.3f} dB")
    print(f"  Sound threshold : {sound_threshold:.3f} dB")

    # Build a balanced Monte-Carlo set of four situations:
    # 0: empty + noise
    # 1: human + noise
    # 2: empty + calling sound
    # 3: human + calling sound
    #
    # Human and sound examples are sampled independently because the two
    # supplied pools are independent datasets.
    n_each = max(1, args.trials // 4)

    scenarios = [
        ("empty + noise", 0, rf_empty, sound_noise),
        ("human + noise", 1, np.concatenate([rf_stand, rf_lying]), sound_noise),
        ("empty + call", 1, rf_empty, np.concatenate([sound_surface, sound_buried])),
        (
            "human + call",
            1,
            np.concatenate([rf_stand, rf_lying]),
            np.concatenate([sound_surface, sound_buried]),
        ),
    ]

    y_true = []
    rf_pred = []
    sound_pred = []
    fusion_pred = []
    fusion_scores = []

    per_scenario = {}

    for name, label, rf_source, sound_source in scenarios:
        scenario_rf = []
        scenario_sound = []
        scenario_fusion = []
        scenario_scores = []

        for _ in range(n_each):
            r = float(rf_source[rng.integers(len(rf_source))])
            s = float(sound_source[rng.integers(len(sound_source))])

            r_z = robust_score(r, rf_med, rf_mad)
            s_z = robust_score(s, sound_med, sound_mad)

            # Positive evidence is clipped at zero. A strong negative
            # deviation therefore does not cancel a positive sensor.
            r_evidence = max(0.0, r_z)
            s_evidence = max(0.0, s_z)

            # Normalize each evidence stream so a large numerical z-score
            # does not dominate the other modality.
            r_unit = r_evidence / (1.0 + r_evidence)
            s_unit = s_evidence / (1.0 + s_evidence)

            score = rf_w * r_unit + sound_w * s_unit

            scenario_rf.append(int(r >= rf_threshold))
            scenario_sound.append(int(s >= sound_threshold))

            # Fusion decision: weighted evidence >= 0.5.
            scenario_fusion.append(int(score >= 0.5))
            scenario_scores.append(score)

            y_true.append(label)
            rf_pred.append(int(r >= rf_threshold))
            sound_pred.append(int(s >= sound_threshold))
            fusion_pred.append(int(score >= 0.5))
            fusion_scores.append(score)

        per_scenario[name] = {
            "truth": label,
            "rf": np.mean(scenario_rf),
            "sound": np.mean(scenario_sound),
            "fusion": np.mean(scenario_fusion),
            "score": np.mean(scenario_scores),
        }

    # Equalize requested trial count approximately.
    actual_trials = len(y_true)

    print(f"\nFusion weights: RF={rf_w:.2f}, Sound={sound_w:.2f}")
    print(f"Generated {actual_trials} balanced simulated trials.")

    print_metrics("RF only", metrics(y_true, rf_pred))
    print_metrics("Sound only", metrics(y_true, sound_pred))
    print_metrics("RF + Sound fusion", metrics(y_true, fusion_pred))

    print("\nDetection rate by scenario")
    print("--------------------------")
    for name, d in per_scenario.items():
        print(
            f"{name:16s} | "
            f"RF {d['rf'] * 100:6.1f}% | "
            f"Sound {d['sound'] * 100:6.1f}% | "
            f"Fusion {d['fusion'] * 100:6.1f}% | "
            f"mean score {d['score']:.3f}"
        )

    # Save results for later plotting/report writing.
    np.savez(
        "fusion_results.npz",
        y_true=np.asarray(y_true),
        rf_pred=np.asarray(rf_pred),
        sound_pred=np.asarray(sound_pred),
        fusion_pred=np.asarray(fusion_pred),
        fusion_score=np.asarray(fusion_scores),
        rf_threshold=rf_threshold,
        sound_threshold=sound_threshold,
        rf_weight=rf_w,
        sound_weight=sound_w,
    )

    print("\nSaved fusion_results.npz")


if __name__ == "__main__":
    main()
