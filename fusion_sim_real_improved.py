
import numpy as np
from pathlib import Path

SEED = 42
N_TRIALS = 5000
CAL_FRAC = 0.50

RF_FILE = "rf_pool.npz"
SOUND_FILE = "sound_pool.npz"
OUT_FILE = "fusion_results_improved.npz"


def robust_median_mad(x):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = np.std(x)
    if scale < 1e-9:
        scale = 1.0
    return med, scale


def ecdf_score(values, negative_reference):
    """Return 0..1 tail score: fraction of negative values <= each value."""
    ref = np.sort(np.asarray(negative_reference, dtype=float))
    values = np.asarray(values, dtype=float)
    return np.searchsorted(ref, values, side="right") / len(ref)


def metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=tp, TN=tn, FP=fp, FN=fn,
                Accuracy=acc, Precision=prec, Recall=rec, FPR=fpr, F1=f1)


def print_metrics(name, m):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"TP={m['TP']} TN={m['TN']} FP={m['FP']} FN={m['FN']}")
    print(f"Accuracy  {m['Accuracy']*100:.2f}%")
    print(f"Precision {m['Precision']*100:.2f}%")
    print(f"Recall    {m['Recall']*100:.2f}%")
    print(f"FPR       {m['FPR']*100:.2f}%")
    print(f"F1        {m['F1']*100:.2f}%")


def trial_value(arr, kind):
    """Collapse 5 decision windows into one trial-level feature."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a
    if kind == "rf":
        # More sensitive than median, but less spike-sensitive than max.
        return np.percentile(a, 75, axis=1)
    if kind == "sound":
        # A call/event may occupy only one of the five 1-s windows.
        return np.max(a, axis=1)
    raise ValueError(kind)


def split_pool(arr, rng):
    arr = np.asarray(arr)
    idx = np.arange(len(arr))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * CAL_FRAC))
    return arr[idx[:cut]], arr[idx[cut:]]


def sample(arr, n, rng):
    return arr[rng.integers(0, len(arr), size=n)]


def make_scenarios(rf, sound, n_each, rng):
    rf_empty, rf_stand, rf_lying, rf_dst, rf_dly = rf
    s_noise, s_surface, s_buried = sound

    # Human = stand/lying. Distance variants are retained in the pool
    # but are not mixed into this primary benchmark unless requested.
    human = np.concatenate([rf_stand, rf_lying], axis=0)
    call = np.concatenate([s_surface, s_buried], axis=0)

    rows = []
    for scenario, rf_pool, snd_pool, label in [
        ("empty + noise", rf_empty, s_noise, 0),
        ("human + noise", human, s_noise, 1),
        ("empty + call", rf_empty, call, 1),
        ("human + call", human, call, 1),
    ]:
        r = sample(rf_pool, n_each, rng)
        s = sample(snd_pool, n_each, rng)
        rows.append((scenario, trial_value(r, "rf"), trial_value(s, "sound"),
                     np.full(n_each, label, dtype=int)))
    return rows


def flatten(rows):
    names, rf, snd, y = [], [], [], []
    for name, r, s, yy in rows:
        names.extend([name] * len(yy))
        rf.append(r); snd.append(s); y.append(yy)
    return np.array(names, dtype=object), np.concatenate(rf), np.concatenate(snd), np.concatenate(y)


def optimize_single(score, y):
    best = None
    for t in np.linspace(0.05, 0.99, 189):
        m = metrics(y, score >= t)
        key = (m["F1"], -m["FPR"])
        if best is None or key > best[0]:
            best = (key, t, m)
    return best[1], best[2]


def optimize_fusion(rf_score, snd_score, y):
    best = None
    for w in np.linspace(0.40, 0.80, 41):
        fused = w * rf_score + (1.0 - w) * snd_score
        for t in np.linspace(0.30, 0.90, 121):
            m = metrics(y, fused >= t)
            key = (m["F1"], -m["FPR"])
            if best is None or key > best[0]:
                best = (key, w, t, m)
    return best[1], best[2], best[3]


# ----------------------------
# Load pools
# ----------------------------
rng = np.random.default_rng(SEED)

rfz = np.load(RF_FILE)
sz = np.load(SOUND_FILE)

print("Loaded pools:")
for k in ["empty", "stand", "lying", "dist_stand", "dist_lying"]:
    print(f"  RF {k:10s}: {len(rfz[k])} trials")
for k in ["noise", "surface", "buried_calling"]:
    print(f"  Sound {k:10s}: {len(sz[k])} trials")

rf_parts = [rfz[k] for k in ["empty", "stand", "lying", "dist_stand", "dist_lying"]]
snd_parts = [sz[k] for k in ["noise", "surface", "buried_calling"]]

rf_split = [split_pool(x, rng) for x in rf_parts]
snd_split = [split_pool(x, rng) for x in snd_parts]

rf_cal = [x[0] for x in rf_split]
rf_eval = [x[1] for x in rf_split]
snd_cal = [x[0] for x in snd_split]
snd_eval = [x[1] for x in snd_split]

# ----------------------------
# Calibration set
# ----------------------------
n_each = N_TRIALS // 4
cal_rows = make_scenarios(rf_cal, snd_cal, n_each, rng)
eval_rows = make_scenarios(rf_eval, snd_eval, n_each, rng)

cal_names, cal_rf, cal_snd, cal_y = flatten(cal_rows)
eval_names, eval_rf, eval_snd, eval_y = flatten(eval_rows)

# Negative reference distributions: empty radar + noise audio.
rf_neg = trial_value(rf_cal[0], "rf")
snd_neg = trial_value(snd_cal[0], "sound")

# New continuous modality scores.
cal_rf_score = ecdf_score(cal_rf, rf_neg)
cal_snd_score = ecdf_score(cal_snd, snd_neg)
eval_rf_score = ecdf_score(eval_rf, rf_neg)
eval_snd_score = ecdf_score(eval_snd, snd_neg)

# Tune only on calibration data.
rf_t, _ = optimize_single(cal_rf_score, cal_y)
snd_t, _ = optimize_single(cal_snd_score, cal_y)
w, fusion_t, _ = optimize_fusion(cal_rf_score, cal_snd_score, cal_y)

# Evaluate on held-out pools.
rf_pred = eval_rf_score >= rf_t
snd_pred = eval_snd_score >= snd_t
fusion_score = w * eval_rf_score + (1.0 - w) * eval_snd_score
fusion_pred = fusion_score >= fusion_t

m_rf = metrics(eval_y, rf_pred)
m_snd = metrics(eval_y, snd_pred)
m_fusion = metrics(eval_y, fusion_pred)

print("\nImproved fusion calibration:")
print(f"RF aggregation       : 75th percentile of 5 windows")
print(f"Audio aggregation    : maximum of 5 windows")
print(f"RF calibration threshold (tail score): {rf_t:.3f}")
print(f"Sound calibration threshold           : {snd_t:.3f}")
print(f"Fusion weight RF/Sound                : {w:.2f}/{1-w:.2f}")
print(f"Fusion threshold                      : {fusion_t:.3f}")
print("Evaluation uses held-out trials (50% calibration / 50% evaluation).")

print_metrics("RF only — held-out evaluation", m_rf)
print_metrics("Sound only — held-out evaluation", m_snd)
print_metrics("RF + Sound fusion — held-out evaluation", m_fusion)

print("\nDetection rate by scenario")
for scenario in ["empty + noise", "human + noise", "empty + call", "human + call"]:
    mask = eval_names == scenario
    print(
        f"{scenario:16s} | "
        f"RF {np.mean(rf_pred[mask])*100:5.1f}% | "
        f"Sound {np.mean(snd_pred[mask])*100:5.1f}% | "
        f"Fusion {np.mean(fusion_pred[mask])*100:5.1f}% | "
        f"mean fusion {np.mean(fusion_score[mask]):.3f}"
    )

np.savez(
    OUT_FILE,
    y=eval_y,
    scenario=eval_names,
    rf_score=eval_rf_score,
    sound_score=eval_snd_score,
    fusion_score=fusion_score,
    rf_pred=rf_pred,
    sound_pred=snd_pred,
    fusion_pred=fusion_pred,
    rf_threshold=rf_t,
    sound_threshold=snd_t,
    fusion_weight_rf=w,
    fusion_weight_sound=1.0-w,
    fusion_threshold=fusion_t,
    seed=SEED,
)

print(f"\nSaved {OUT_FILE}")
