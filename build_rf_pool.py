"""
build_rf_pool.py  -  REAL-RADAR model of the beacon's RF/Doppler (breathing) sensor
===================================================================================
Data: "Signs of life detection behind obstacles using a UWB radar sensor"
      Uzunidis, Kasnesis, Patrikakis, Mitilineos (Univ. of West Attica), Zenodo 10.5281/zenodo.10779256
      Novelda X4M200 UWB radar, 17 samples/s, range bins 0.48-6 m (step ~0.05144 m)
      standing humans behind 30 cm wall, lying humans behind 20 cm wall, 1-4 m, plus EMPTY sessions.

HOW TO USE (on your own machine - the 744 MB zip could not be downloaded in my sandbox):
  1. Download "Data for Zenodo_v2.zip" from https://zenodo.org/records/10779256 and unzip it.
  2. python build_rf_pool.py  <unzipped_folder>
  3. Check the printed file classification. If files land in "UNCLASSIFIED", edit the
     keyword lists below to match the real folder/file names, and re-run.
  4. Copy rf_pool.npz next to fusion_sim_real.py and run that.

Per CSV (first row = range axis in metres, later rows = time samples):
  window = 20 s (340 samples), step 5 s, 5 consecutive windows = one trial
  1. subtract per-bin time mean (static clutter / wall removal)
  2. per range bin: PSD; breathing-band power (0.15-0.60 Hz) / reference-band power (2.5-6 Hz)
  3. statistic = 10*log10( max over range gate 0.8-4.5 m )   -> raw dB value
Same statistic for human and empty sessions, so the max-over-bins bias hits both equally.

NOTE: written from the dataset description; tested only on mock files in the documented format.
"""
import sys, re, glob, os, numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
# ---- edit these keyword lists if your folder / file names differ ---------------------------
EMPTY_KW = ["empty", "no human", "nohuman", "no_human", "without", "background", "noone", "no-human", "norm"]
STAND_KW = ["stand"]
LYING_KW = ["lying", "lay", "lie", "supine"]
EXCLUDE_KW = ["overlap"]            # overlapped 20 s copies - we build our own windows
# ----------------------------------------------------------------------------------------
FS, WIN_S, STEP_S, NWIN = 17.0, 20, 5, 5
GATE = (0.8, 4.5)
BREATH = (0.15, 0.60)
REF = (2.5, 6.0)

def classify(path):
    p = path.lower()
    if any(k in p for k in EXCLUDE_KW):
        return None
    has_h = None
    if any(k in p for k in EMPTY_KW):
        return "empty"
    if any(k in p for k in LYING_KW):
        return "lying"
    if any(k in p for k in STAND_KW):
        return "stand"
    return "UNCLASSIFIED"

def stat_window(X, rng_axis):
    X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    nfft = 4096
    spec = np.abs(np.fft.rfft(X * np.hanning(n)[:, None], n=nfft, axis=0)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / FS)
    gate = (rng_axis >= GATE[0]) & (rng_axis <= GATE[1])
    b = spec[(f >= BREATH[0]) & (f <= BREATH[1])][:, gate].mean(axis=0)
    r = spec[(f >= REF[0]) & (f <= REF[1])][:, gate].mean(axis=0)
    return 10 * np.log10(np.max(b / (r + 1e-18)) + 1e-18)

def process(path):
    A = np.loadtxt(path, delimiter=",")
    rng_axis, X = A[0], A[1:]
    w, s = int(WIN_S * FS), int(STEP_S * FS)
    starts = list(range(0, X.shape[0] - w + 1, s))
    vals = [stat_window(X[a:a + w], rng_axis) for a in starts]
    return [vals[i:i + NWIN] for i in range(0, len(vals) - NWIN + 1)]

files = [f for f in glob.glob(os.path.join(ROOT, "**", "*.csv"), recursive=True)]
cls = {f: classify(f) for f in files}
cls = {f: c for f, c in cls.items() if c is not None}
print("Found", len(cls), "CSV files")
for c in ["empty", "stand", "lying", "UNCLASSIFIED"]:
    fs = [f for f, k in cls.items() if k == c]
    print(f"  {c:13s}{len(fs):5d}   e.g. {[os.path.relpath(x, ROOT) for x in fs[:2]]}")
if any(v == "UNCLASSIFIED" for v in cls.values()):
    sys.exit("Fix keyword lists at the top of this script, then re-run.")

pool = {"empty": [], "stand": [], "lying": []}
dist = {"stand": [], "lying": []}
for f, c in cls.items():
    trials = process(f)
    pool[c] += trials
    if c != "empty":
        m = re.search(r"(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)", os.path.basename(f))   # "<dist>_<breath rate>"
        dist[c] += [float(m.group(1)) if m else np.nan] * len(trials)
out = {k: np.array(v) for k, v in pool.items()}
out.update({"dist_stand": np.array(dist["stand"]), "dist_lying": np.array(dist["lying"])})
for k in pool:
    v = out[k]
    print(f"{k:6s} trials {v.shape[0]:5d}  median {np.median(v):6.1f} dB   p10 {np.percentile(v,10):6.1f}   p90 {np.percentile(v,90):6.1f}")
np.savez("rf_pool.npz", **out)
print("saved rf_pool.npz")
