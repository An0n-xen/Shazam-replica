import logging
import warnings
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.io import wavfile

try:
    # running from main.py
    from utils.config import AudioConfig as Config
    from utils.logging_config import setup_logger
    from utils.visualize import visualize_constellation_map
except ImportError:
    # running from this file
    from config import AudioConfig as Config
    from logging_config import setup_logger
    from visualize import visualize_constellation_map

warnings.filterwarnings("ignore")

logger = setup_logger(__name__, level=logging.INFO)


def load_audio(filepath):
    """
    Load audio file and convert to mono at target sample rate.

    Args:
        filepath: Path to audio file (wav, mp3, etc.)

    Returns:
        audio: numpy array of audio samples
        sr: sample rate
    """
    try:
        # Try librosa first (handles more formats)
        import librosa

        audio, sr = librosa.load(filepath, sr=Config.SAMPLE_RATE, mono=True)
        logger.info(f"✓ Loaded with librosa: {filepath}")

    except ImportError:
        # Fall back to scipy (wav only)
        sr, audio = wavfile.read(filepath)

        # Convert to mono if stereo
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        # Convert to float
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Resample if needed
        if sr != Config.SAMPLE_RATE:
            logger.warning(
                f"Warning: Sample rate is {sr}, expected {Config.SAMPLE_RATE}"
            )

        logger.info(f"✓ Loaded with scipy: {filepath}")

    duration = len(audio) / sr
    logger.info(f"  Duration: {duration:.2f} seconds")
    logger.info(f"  Sample rate: {sr} Hz")
    logger.info(f"  Samples: {len(audio)}")

    return audio, sr


def generate_spectrogram(audio, sr):
    """
    Generate spectrogram from audio using STFT.

    Args:
        audio: Audio signal
        sr: Sample rate

    Returns:
        freqs: Frequency bins
        times: Time bins
        spec: Spectrogram (2D array, already in log scale)
    """
    # Calculate parameters
    nperseg = Config.FFT_WINDOW_SIZE
    noverlap = int(nperseg * Config.OVERLAP_RATIO)

    # Compute spectrogram
    freqs, times, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
    )

    # Convert to log scale (dB) for better peak detection
    # Add small epsilon to avoid log(0)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)

    logger.info(f"\n✓ Spectrogram generated:")
    logger.info(f"  Shape: {Sxx_log.shape} (freq bins × time bins)")
    logger.info(f"  Time bins: {len(times)}")
    logger.info(f"  Frequency bins: {len(freqs)}")
    logger.info(f"  Max frequency: {freqs[-1]:.0f} Hz")

    return freqs, times, Sxx_log


def find_peaks(spec, freqs, times):
    """
    Find local maxima (peaks) in the spectrogram.
    This creates the "constellation map" - the sparse set of points.

    Args:
        spec: Spectrogram (2D array)
        freqs: Frequency bins
        times: Time bins

    Returns:
        peaks: List of (time_idx, freq_idx) tuples
    """
    # Step 1: Find local maxima using maximum filter
    struct_size = Config.PEAK_NEIGHBORHOOD_SIZE
    local_max = maximum_filter(spec, size=(struct_size, struct_size))

    # A peak is where the value equals the local maximum
    peak_mask = spec == local_max

    # Step 2: Apply amplitude threshold
    # Only keep peaks above certain percentile
    threshold = np.percentile(spec, Config.MIN_AMPLITUDE_PERCENTILE)
    peak_mask = peak_mask & (spec > threshold)

    # Get coordinates of all peaks
    peak_coords = np.argwhere(peak_mask)

    logger.info(f"\n✓ Initial peaks found: {len(peak_coords)}")

    # Step 3: Apply density control - limit peaks per time window
    # This ensures uniform coverage (paper's "density criterion")
    peaks = apply_density_control(peak_coords, spec, times)

    logger.info(f"✓ After density control: {len(peaks)} peaks")
    logger.info(f"  Average: {len(peaks) / times[-1]:.1f} peaks/second")

    return peaks


def apply_density_control(peak_coords, spec, times):
    """
    Limit the number of peaks per time window to ensure uniform coverage.
    Keep only the highest amplitude peaks in each window.

    Args:
        peak_coords: Array of (freq_idx, time_idx) coordinates
        spec: Spectrogram for amplitude values
        times: Time bins

    Returns:
        filtered_peaks: List of (time_idx, freq_idx) tuples
    """
    # Convert time window to bins
    time_resolution = times[1] - times[0]
    window_bins = int(Config.TIME_WINDOW_SEC / time_resolution)

    # Group peaks by time window
    max_time_idx = spec.shape[1]
    filtered_peaks = []

    for window_start in range(0, max_time_idx, window_bins):
        window_end = min(window_start + window_bins, max_time_idx)

        # Find peaks in this window
        window_peaks = peak_coords[
            (peak_coords[:, 1] >= window_start) & (peak_coords[:, 1] < window_end)
        ]

        if len(window_peaks) == 0:
            continue

        # Get amplitudes for these peaks
        amplitudes = [spec[freq_idx, time_idx] for freq_idx, time_idx in window_peaks]

        # Keep only top N peaks by amplitude
        n_keep = min(Config.PEAKS_PER_TIME_WINDOW, len(window_peaks))
        top_indices = np.argsort(amplitudes)[-n_keep:]

        # Store as (time_idx, freq_idx) for consistency
        for idx in top_indices:
            freq_idx, time_idx = window_peaks[idx]
            filtered_peaks.append((time_idx, freq_idx))

    return filtered_peaks


def create_constellation_map(audio_path, visualize=True, save_plot=None):
    """
    Complete pipeline: Load audio → Generate spectrogram → Find peaks

    Args:
        audio_path: Path to audio file
        visualize: Whether to show plots
        save_plot: Path to save visualization (optional)

    Returns:
        peaks: List of (time_idx, freq_idx) tuples
        freqs: Frequency bins
        times: Time bins
        spec: Spectrogram
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 2: Creating Constellation Map")
    logger.info(f"{'='*60}")

    # Step 1: Load audio
    logger.info("\n[1/3] Loading audio...")
    audio, sr = load_audio(audio_path)

    # Step 2: Generate spectrogram
    logger.info("\n[2/3] Generating spectrogram...")
    freqs, times, spec = generate_spectrogram(audio, sr)

    # Step 3: Find peaks
    logger.info("\n[3/3] Detecting peaks...")
    peaks = find_peaks(spec, freqs, times)

    # Visualize results
    if visualize:
        logger.info("\n[Visualizing] Generating constellation map plot...")
        visualize_constellation_map(
            audio, sr, spec, freqs, times, peaks, save_path=save_plot
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Constellation map created successfully!")
    logger.info(f"  Total peaks: {len(peaks)}")
    logger.info(f"  Duration: {times[-1]:.2f} seconds")
    logger.info(f"  Peak density: {len(peaks)/times[-1]:.1f} peaks/second")
    logger.info(f"{'='*60}\n")

    return peaks, freqs, times, spec


if __name__ == "__main__":
    """
    Test the constellation map generation on a sample audio file.

    """

    AUDIO_FILE = "./data/db_tracks/sample1.mp3"
    SAVE_PLOT = "constellation_map.png"

    peaks, freqs, times, spec = create_constellation_map(
        AUDIO_FILE, visualize=True, save_plot=SAVE_PLOT
    )
