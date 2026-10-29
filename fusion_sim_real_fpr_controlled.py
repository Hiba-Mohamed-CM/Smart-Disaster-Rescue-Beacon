
import numpy as np

SEED = 42
N_TRIALS = 5000
CAL_FRAC = 0.50
TARGET_FPR = 0.05

RF_FILE = "rf_pool.npz"
SOUND_FILE = "sound_pool.npz"
OUT_FILE = "fusion_results_fpr_controlled.npz"


def metrics(y, pred):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
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


def split_pool(arr, rng):
    arr = np.asarray(arr)
    idx = np.arange(len(arr))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * CAL_FRAC))
    return arr[idx[:cut]], arr[idx[cut:]]


def sample(arr, n, rng):
    return arr[rng.integers(0, len(arr), size=n)]


def aggregate(arr, kind):
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a
    if kind == "rf":
        return np.percentile(a, 75, axis=1)
    return np.max(a, axis=1)


def make_scenarios(rf, snd, n_each, rng):
    empty, stand, lying, _, _ = rf
    noise, surface, buried = snd
    human = np.concatenate([stand, lying], axis=0)
    call = np.concatenate([surface, buried], axis=0)

    rows = []
    specs = [
        ("empty + noise", empty, noise, 0),
        ("human + noise", human, noise, 1),
        ("empty + call", empty, call, 1),
        ("human + call", human, call, 1),
    ]
    for name, rp, sp, label in specs:
        r = aggregate(sample(rp, n_each, rng), "rf")
        s = aggregate(sample(sp, n_each, rng), "sound")
        rows.append((name, r, s, np.full(n_each, label, dtype=int)))
    return rows


def flatten(rows):
    names, rf, snd, y = [], [], [], []
    for name, r, s, yy in rows:
        names.extend([name] * len(yy))
        rf.append(r)
        snd.append(s)
        y.append(yy)
    return np.array(names, dtype=object), np.concatenate(rf), np.concatenate(snd), np.concatenate(y)


def percentile_score(values, negative_reference):
    ref = np.sort(np.asarray(negative_reference, dtype=float))
    return np.searchsorted(ref, values, side="right") / len(ref)


def threshold_at_fpr(scores, y, target=0.05):
    """Choose the lowest threshold whose negative calibration FPR is <= target."""
    neg = np.asarray(scores)[np.asarray(y) == 0]
    if len(neg) == 0:
        return 1.0
    # Quantile with 'higher' makes the threshold conservative.
    return float(np.quantile(neg, 1.0 - target, method="higher"))


def optimize_fusion_with_fpr(rf_score, snd_score, y, target=0.05):
    best = None
    for w in np.linspace(0.40, 0.80, 41):
        fused = w * rf_score + (1 - w) * snd_score
        # Threshold candidates based on calibration negative distribution.
        neg = fused[np.asarray(y) == 0]
        if len(neg) == 0:
            continue
        candidates = np.unique(np.quantile(
            neg, np.linspace(0.90, 1.00, 101), method="higher"
        ))
        for t in candidates:
            m = metrics(y, fused >= t)
            if m["FPR"] <= target + 1e-12:
                # Primary objective: recall, then F1, then lower FPR.
                key = (m["Recall"], m["F1"], -m["FPR"])
                if best is None or key > best[0]:
                    best = (key, w, float(t), m)

    if best is None:
        # Fallback: strictest possible threshold.
        fused = 0.5 * rf_score + 0.5 * snd_score
        t = float(np.max(fused[np.asarray(y) == 0]))
        return 0.5, t, metrics(y, fused >= t)
    return best[1], best[2], best[3]


rng = np.random.default_rng(SEED)

rfz = np.load(RF_FILE)
sz = np.load(SOUND_FILE)

print("Loaded pools:")
for k in ["empty", "stand", "lying", "dist_stand", "dist_lying"]:
    print(f"  RF {k:10s}: {len(rfz[k])} trials")
for k in ["noise", "surface", "buried_calling"]:
    print(f"  Sound {k:14s}: {len(sz[k])} trials")

rf_parts = [rfz[k] for k in ["empty", "stand", "lying", "dist_stand", "dist_lying"]]
snd_parts = [sz[k] for k in ["noise", "surface", "buried_calling"]]

rf_split = [split_pool(x, rng) for x in rf_parts]
snd_split = [split_pool(x, rng) for x in snd_parts]

rf_cal = [x[0] for x in rf_split]
rf_eval = [x[1] for x in rf_split]
snd_cal = [x[0] for x in snd_split]
snd_eval = [x[1] for x in snd_split]

n_each = N_TRIALS // 4
cal_rows = make_scenarios(rf_cal, snd_cal, n_each, rng)
eval_rows = make_scenarios(rf_eval, snd_eval, n_each, rng)

cal_names, cal_rf, cal_snd, cal_y = flatten(cal_rows)
eval_names, eval_rf, eval_snd, eval_y = flatten(eval_rows)

# Calibrate each modality against genuine negative conditions.
rf_negative = aggregate(rf_cal[0], "rf")
sound_negative = aggregate(snd_cal[0], "sound")

cal_rf_score = percentile_score(cal_rf, rf_negative)
cal_snd_score = percentile_score(cal_snd, sound_negative)
eval_rf_score = percentile_score(eval_rf, rf_negative)
eval_snd_score = percentile_score(eval_snd, sound_negative)

# Find thresholds constrained by false-positive rate.
rf_t = threshold_at_fpr(cal_rf_score, cal_y, TARGET_FPR)
snd_t = threshold_at_fpr(cal_snd_score, cal_y, TARGET_FPR)
w, fusion_t, cal_fusion_metrics = optimize_fusion_with_fpr(
    cal_rf_score, cal_snd_score, cal_y, TARGET_FPR
)

# Held-out evaluation.
rf_pred = eval_rf_score >= rf_t
snd_pred = eval_snd_score >= snd_t
fusion_score = w * eval_rf_score + (1 - w) * eval_snd_score
fusion_pred = fusion_score >= fusion_t

m_rf = metrics(eval_y, rf_pred)
m_snd = metrics(eval_y, snd_pred)
m_fusion = metrics(eval_y, fusion_pred)

print("\nFPR-controlled calibration:")
print(f"Target maximum calibration FPR : {TARGET_FPR*100:.1f}%")
print("RF aggregation                  : 75th percentile of 5 windows")
print("Audio aggregation               : maximum of 5 windows")
print(f"RF threshold                    : {rf_t:.3f}")
print(f"Sound threshold                 : {snd_t:.3f}")
print(f"Fusion weight RF/Sound          : {w:.2f}/{1-w:.2f}")
print(f"Fusion threshold                : {fusion_t:.3f}")
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
    fusion_weight_sound=1-w,
    fusion_threshold=fusion_t,
    target_fpr=TARGET_FPR,
    seed=SEED,
)

print(f"\nSaved {OUT_FILE}")
