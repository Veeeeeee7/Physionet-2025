import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
import pywt
from scipy.fft import fft, fftfreq

def compute_r_peaks(ecg_signal, fs, distance_seconds=0.2, height=None):
    """
    Computes R peaks in an ECG signal using a simple peak-finding method.
    
    Parameters:
    - ecg_signal: 1D numpy array, the (filtered) ECG signal.
    - fs: float, the sampling frequency (Hz).
    - distance_seconds: float, minimum time interval between successive peaks.
    - height: float or None, the minimum height for a peak (if provided).
    
    Returns:
    - peaks: numpy array of indices corresponding to the R peaks.
    """
    distance_samples = int(distance_seconds * fs)
    peaks, _ = find_peaks(ecg_signal, distance=distance_samples, height=height)
    return peaks

def compute_rr_intervals(r_peaks, fs):
    """
    Computes RR intervals (the time differences between successive R peaks).
    
    Parameters:
    - r_peaks: numpy array of indices of R peaks.
    - fs: float, the sampling frequency (Hz).
    
    Returns:
    - rr_intervals_ms: numpy array of RR intervals in milliseconds.
    """
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    rr_intervals_ms = rr_intervals * 1000  # convert to milliseconds
    return rr_intervals_ms

def compute_hrv(rr_intervals_ms):
    """
    Computes heart rate variability metrics from RR intervals.
    
    Parameters:
    - rr_intervals_ms: numpy array of RR intervals in milliseconds.
    
    Returns:
    - sdnn: float, standard deviation of RR intervals.
    - rmssd: float, root mean square of successive differences.
    """
    sdnn = np.std(rr_intervals_ms, ddof=1)  # Standard Deviation of NN intervals
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals_ms)**2))
    return sdnn, rmssd

def compute_fft(ecg_signal, fs):
    """
    Computes the FFT of an ECG signal and returns the frequency bins and corresponding amplitudes.
    
    Parameters:
    - ecg_signal: 1D numpy array, the ECG signal.
    - fs: float, the sampling frequency (Hz).
    
    Returns:
    - freqs: numpy array, the frequencies corresponding to the FFT bins (only positive frequencies).
    - amplitudes: numpy array, the amplitude spectrum of the ECG signal.
    """
    if len(ecg_signal) > 200:
        indices = np.linspace(0, len(ecg_signal) - 1, 200, dtype=int)
        ecg_signal = ecg_signal[indices]
    
    N = len(ecg_signal)
    # Compute the FFT of the signal
    fft_vals = fft(ecg_signal)
    # Compute the corresponding frequency bins
    fft_freqs = fftfreq(N, d=1/fs)
    
    # Only keep the positive frequencies (since the FFT is symmetric for real-valued signals)
    pos_mask = fft_freqs >= 0
    freqs = fft_freqs[pos_mask]
    
    # Normalize and obtain the amplitude spectrum
    # Multiplying by 2 compensates for the single-sided spectrum (except for DC component)
    amplitudes = (2.0 / N) * np.abs(fft_vals[pos_mask])
    
    # pad the arrays to ensure they are of length 100
    if len(freqs) < 100:
        freqs = np.pad(freqs, (0, 100 - len(freqs)), mode='constant')
    if len(amplitudes) < 100:
        amplitudes = np.pad(amplitudes, (0, 100 - len(amplitudes)), mode='constant')

    return freqs, amplitudes

def compute_wavelet_energy(ecg_signal, wavelet='db4', level=4):
    """
    Computes the energy of detail coefficients from a discrete wavelet transform.
    
    Parameters:
    - ecg_signal: 1D numpy array, the ECG signal.
    - wavelet: string, type of wavelet to use.
    - level: int, number of decomposition levels.
    
    Returns:
    - energies: list containing the energy of each detail level.
    """
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    energies = []
    # Skip the approximation coefficient (coeffs[0]) and compute energies for details
    for detail in coeffs[1:]:
        energies.append(np.sum(np.square(detail)))
    return energies
