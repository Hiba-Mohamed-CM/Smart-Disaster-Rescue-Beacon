"""
build_sound_pool.py  -  REAL-AUDIO model of the beacon's sound sensor
=====================================================================
Data: ESC-50 (Piczak 2015), https://github.com/karolpiczak/ESC-50  (CC BY-NC)
  Victim-like sounds : crying_baby, coughing, door_wood_knock   (proxy: ESC-50 has no screams / SOS calls)
  Ambient / nuisance : wind, rain, thunderstorm, engine, helicopter, chainsaw,
                       crackling_fire, footsteps

Per trial = 5 s = 5 consecutive 1 s decision windows.
  trial = ambient clip  [+ victim clip scaled to a chosen SNR vs the ambient]
Feature per window ("event contrast"):
  300-3400 Hz band energy per 46 ms frame (dB); contrast = max(frame) - median(frame)
  -> steady wind/engine ~ low contrast, calls/knocks/thunder bursts ~ high contrast.

The ONLY assumed numbers are the victim-to-ambient SNRs below (how much rubble/soil/
distance attenuates a voice). Everything else comes from real recordings.

Usage:  python build_sound_pool.py  <path-to-folder-with-ESC-50-wavs>  <path-to-esc50.csv>
Output: sound_pool.npz
"""
import sys, csv, numpy as np
from scipy.io import wavfile

WAV_DIR = sys.argv[1] if len(sys.argv) > 1 else "wav"
META = sys.argv[2] if len(sys.argv) > 2 else "esc50.csv"
VICTIM = {"crying_baby", "coughing", "door_wood_knock"}
NOISE = {"wind", "rain", "thunderstorm", "engine", "helicopter", "chainsaw", "crackling_fire", "footsteps"}
SNR_SURFACE_DB = +6      # victim 6 dB above ambient (open air, near the beacon)
SNR_BURIED_DB = -9       # victim 9 dB below ambient (attenuated by rubble/soil)
N_TRIALS = 3000
MIC_FLOOR_RMS = 1e-3     # microphone self-noise (~ -60 dBFS)
FS = 44100
FRAME = 2048
rng = np.random.default_rng(1)

def load(name):
    fs, x = wavfile.read(f"{WAV_DIR}/{name}")
    assert fs == FS
    return x.astype(np.float32) / 32768.0

clips = {"victim": [], "noise": []}
for r in csv.DictReader(open(META)):
    kind = "victim" if r["category"] in VICTIM else "noise" if r["category"] in NOISE else None
    if kind:
        try:
            clips[kind].append(load(r["filename"]))
        except FileNotFoundError:
            pass
print({k: len(v) for k, v in clips.items()})

def active_rms(x):
    f = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
    e = (f ** 2).mean(1)
    act = e > 0.01 * e.max() if e.max() > 0 else e > 0
    return np.sqrt(e[act].mean()) if act.any() else 1e-9

freqs = np.fft.rfftfreq(FRAME, 1 / FS)
band = (freqs >= 300) & (freqs <= 3400)
win = np.hanning(FRAME)

def contrast_windows(x):
    """x: 5 s signal -> contrast (dB) for each of 5 one-second windows."""
    out = []
    for w in range(5):
        seg = x[w * FS:(w + 1) * FS]
        f = seg[: len(seg) // FRAME * FRAME].reshape(-1, FRAME) * win
        e = (np.abs(np.fft.rfft(f, axis=1)[:, band]) ** 2).sum(1)
        db = 10 * np.log10(e + 1e-12)
        out.append(db.max() - np.median(db))
    return out

def trial(with_victim, snr_db):
    n = clips["noise"][rng.integers(len(clips["noise"]))]
    x = n.copy()
    if with_victim:
        v = clips["victim"][rng.integers(len(clips["victim"]))]
        v = np.roll(v, rng.integers(len(v)))                 # random start time
        g = active_rms(n) * 10 ** (snr_db / 20) / active_rms(v)
        x = x + g * v
    x = x + rng.normal(0, MIC_FLOOR_RMS, len(x)).astype(np.float32)
    return contrast_windows(x)

pool = {
    "noise": np.array([trial(False, 0) for _ in range(N_TRIALS)]),
    "surface": np.array([trial(True, SNR_SURFACE_DB) for _ in range(N_TRIALS)]),
    "buried_calling": np.array([trial(True, SNR_BURIED_DB) for _ in range(N_TRIALS)]),
}
for k, v in pool.items():
    print(f"{k:15s} contrast dB: median {np.median(v):5.1f}  p10 {np.percentile(v,10):5.1f}  p90 {np.percentile(v,90):5.1f}")
np.savez("sound_pool.npz", **pool, snr_surface=SNR_SURFACE_DB, snr_buried=SNR_BURIED_DB)
print("saved sound_pool.npz", {k: v.shape for k, v in pool.items()})
