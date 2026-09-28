import numpy as np
from scipy import signal


def rmse(x, y):
    return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))


def nrmse_range(x, y):
    x = np.asarray(x)
    return rmse(x, y) / (float(np.max(x) - np.min(x)) + 1e-12)


def prd(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    den = np.sum((x - np.mean(x)) ** 2)
    return float(100 * np.sqrt(np.sum((x - y) ** 2) / (den + 1e-12)))


def correlation(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def estimate_ecg_hr(x, fs):
    x = np.asarray(x, dtype=float)
    if len(x) < fs:
        return np.nan
    nyq = fs / 2
    b, a = signal.butter(2, [5 / nyq, min(25 / nyq, 0.99)], btype="bandpass")
    xf = signal.filtfilt(b, a, x)
    z = (xf - np.median(xf)) / (np.std(xf) + 1e-12)
    peaks, _ = signal.find_peaks(z, distance=int(0.30 * fs), prominence=0.7)
    if len(peaks) < 2:
        peaks, _ = signal.find_peaks(-z, distance=int(0.30 * fs), prominence=0.7)
    if len(peaks) < 2:
        return np.nan
    duration = (peaks[-1] - peaks[0]) / fs
    return float(60 * (len(peaks) - 1) / (duration + 1e-12))


def eda_peak_count(x, fs):
    x = np.asarray(x, dtype=float)
    cutoff = min(1.0, 0.45 * fs)
    if fs > 4:
        b, a = signal.butter(2, cutoff / (fs / 2), btype="low")
        xf = signal.filtfilt(b, a, x)
    else:
        xf = x
    prominence = max(0.05 * np.std(xf), 1e-9)
    peaks, _ = signal.find_peaks(xf, distance=max(1, int(fs)), prominence=prominence)
    return int(len(peaks))


def ppg_peak_count(x, fs):
    x = np.asarray(x, dtype=float)
    nyq = fs / 2
    lo = max(0.35 / nyq, 1e-5)
    hi = min(5.0 / nyq, 0.99)
    if lo >= hi:
        xf = signal.detrend(x)
    else:
        b, a = signal.butter(2, [lo, hi], btype="bandpass")
        xf = signal.filtfilt(b, a, x)
    z = (xf - np.median(xf)) / (np.std(xf) + 1e-12)
    peaks, _ = signal.find_peaks(z, distance=max(1, int(0.30 * fs)), prominence=0.35)
    return int(len(peaks))


def resp_peak_count(x, fs):
    x = np.asarray(x, dtype=float)
    nyq = fs / 2
    lo = max(0.05 / nyq, 1e-5)
    hi = min(0.8 / nyq, 0.99)
    if lo >= hi:
        xf = signal.detrend(x)
    else:
        b, a = signal.butter(2, [lo, hi], btype="bandpass")
        xf = signal.filtfilt(b, a, x)
    z = (xf - np.median(xf)) / (np.std(xf) + 1e-12)
    peaks, _ = signal.find_peaks(z, distance=max(1, int(fs)), prominence=0.20)
    return int(len(peaks))


def physiology_error(modality, raw, recon, fs):
    if modality == "ECG":
        a = estimate_ecg_hr(raw, fs)
        b = estimate_ecg_hr(recon, fs)
        return "HR_abs_error_bpm", abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan, a, b
    if modality == "EDA":
        a, b = eda_peak_count(raw, fs), eda_peak_count(recon, fs)
        return "EDA_peak_count_abs_error", abs(a - b), a, b
    if modality == "PPG":
        a, b = ppg_peak_count(raw, fs), ppg_peak_count(recon, fs)
        return "PPG_pulse_count_abs_error", abs(a - b), a, b
    if modality == "RESP":
        a, b = resp_peak_count(raw, fs), resp_peak_count(recon, fs)
        return "RESP_cycle_count_abs_error", abs(a - b), a, b
    return "none", np.nan, np.nan, np.nan
