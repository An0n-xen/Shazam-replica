"""
Phase 2: Spectrogram Generation and Peak Detection (Constellation Map)
This implements the "star field" generation from the Shazam paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore")


# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters for fingerprinting"""

    # Audio processing
    SAMPLE_RATE = 22050  # Hz (will resample to this)

    # Spectrogram parameters
    FFT_WINDOW_SIZE = 4096  # Number of samples per FFT
    OVERLAP_RATIO = 0.5  # 50% overlap between windows

    # Peak detection parameters
    PEAK_NEIGHBORHOOD_SIZE = 20  # Local maximum filter size
    MIN_AMPLITUDE_PERCENTILE = 80  # Keep peaks above this percentile
    PEAKS_PER_TIME_WINDOW = 5  # Max peaks per time window
    TIME_WINDOW_SEC = 0.5  # Time window for peak limiting


# ==================== AUDIO LOADING ====================
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
        print(f"✓ Loaded with librosa: {filepath}")

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
            print(f"Warning: Sample rate is {sr}, expected {Config.SAMPLE_RATE}")
            print("Install librosa for automatic resampling: pip install librosa")

        print(f"✓ Loaded with scipy: {filepath}")

    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Samples: {len(audio)}")

    return audio, sr


# ==================== SPECTROGRAM GENERATION ====================
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

    print(f"\n✓ Spectrogram generated:")
    print(f"  Shape: {Sxx_log.shape} (freq bins × time bins)")
    print(f"  Time bins: {len(times)}")
    print(f"  Frequency bins: {len(freqs)}")
    print(f"  Max frequency: {freqs[-1]:.0f} Hz")

    return freqs, times, Sxx_log


# ==================== PEAK DETECTION ====================
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

    print(f"\n✓ Initial peaks found: {len(peak_coords)}")

    # Step 3: Apply density control - limit peaks per time window
    # This ensures uniform coverage (paper's "density criterion")
    peaks = apply_density_control(peak_coords, spec, times)

    print(f"✓ After density control: {len(peaks)} peaks")
    print(f"  Average: {len(peaks) / times[-1]:.1f} peaks/second")

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


# ==================== VISUALIZATION ====================
def visualize_constellation_map(audio, sr, spec, freqs, times, peaks, save_path=None):
    """
    Visualize the constellation map (peaks on spectrogram).
    This should look like a "star field".

    Args:
        audio: Original audio signal
        sr: Sample rate
        spec: Spectrogram
        freqs: Frequency bins
        times: Time bins
        peaks: List of (time_idx, freq_idx) peak coordinates
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Waveform
    ax1 = axes[0]
    time_audio = np.arange(len(audio)) / sr
    ax1.plot(time_audio, audio, linewidth=0.5, color="steelblue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Audio Waveform")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Full Spectrogram
    ax2 = axes[1]
    im = ax2.pcolormesh(times, freqs, spec, shading="auto", cmap="viridis")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Spectrogram (Log Scale)")
    plt.colorbar(im, ax=ax2, label="Power (dB)")

    # Plot 3: Constellation Map (peaks only)
    ax3 = axes[2]
    # Show spectrogram faintly in background
    ax3.pcolormesh(
        times,
        freqs,
        spec,
        shading="auto",
        cmap="gray",
        alpha=0.3,
        vmin=spec.min(),
        vmax=spec.max(),
    )

    # Plot peaks as red dots (the "stars")
    if peaks:
        peak_times = [times[t_idx] for t_idx, f_idx in peaks]
        peak_freqs = [freqs[f_idx] for t_idx, f_idx in peaks]
        ax3.scatter(
            peak_times, peak_freqs, c="red", s=5, alpha=0.8, label=f"{len(peaks)} peaks"
        )

    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_xlabel("Time (s)")
    ax3.set_title('Constellation Map ("Star Field")')
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Visualization saved to: {save_path}")

    plt.show()


# ==================== MAIN PIPELINE ====================
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
    print(f"\n{'='*60}")
    print(f"PHASE 2: Creating Constellation Map")
    print(f"{'='*60}")

    # Step 1: Load audio
    print("\n[1/3] Loading audio...")
    audio, sr = load_audio(audio_path)

    # Step 2: Generate spectrogram
    print("\n[2/3] Generating spectrogram...")
    freqs, times, spec = generate_spectrogram(audio, sr)

    # Step 3: Find peaks
    print("\n[3/3] Detecting peaks...")
    peaks = find_peaks(spec, freqs, times)

    # Visualize results
    if visualize:
        print("\n[Visualizing] Generating constellation map plot...")
        visualize_constellation_map(
            audio, sr, spec, freqs, times, peaks, save_path=save_plot
        )

    print(f"\n{'='*60}")
    print(f"✓ Constellation map created successfully!")
    print(f"  Total peaks: {len(peaks)}")
    print(f"  Duration: {times[-1]:.2f} seconds")
    print(f"  Peak density: {len(peaks)/times[-1]:.1f} peaks/second")
    print(f"{'='*60}\n")

    return peaks, freqs, times, spec


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    """
    Test the constellation map generation on a sample audio file.

    Usage:
        python phase2_constellation.py

    Before running:
        1. Create a 'data' folder
        2. Add a test audio file (e.g., 'test.wav' or 'test.mp3')
        3. Update the file path below
    """

    # === CONFIGURATION ===
    # Update this path to your audio file
    AUDIO_FILE = "./data/db_tracks/sample1.mp3"  # Change this to your file!
    SAVE_PLOT = "constellation_map.png"  # Where to save visualization

    print(
        """
    ╔════════════════════════════════════════════════════════════╗
    ║  Phase 2: Spectrogram & Peak Detection (Constellation Map) ║
    ╚════════════════════════════════════════════════════════════╝
    
    This script will:
    1. Load your audio file
    2. Generate a spectrogram
    3. Detect peaks (the "constellation map")
    4. Visualize the results
    
    Make sure you have an audio file ready!
    """
    )

    try:
        # Run the pipeline
        peaks, freqs, times, spec = create_constellation_map(
            AUDIO_FILE, visualize=True, save_plot=SAVE_PLOT
        )

        print("\n✅ SUCCESS! Your constellation map is ready.")
        print("\nWhat to look for in the visualization:")
        print("  • Bottom plot should show red dots like a 'star field'")
        print("  • Peaks should be distributed across time (not clustered)")
        print("  • Typical density: 3-10 peaks per second")
        print("\nNext steps:")
        print("  • Try different audio files")
        print("  • Adjust Config parameters if needed")
        print("  • Move to Phase 3: Combinatorial Hashing")

    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find audio file: {AUDIO_FILE}")
        print("\nPlease:")
        print("  1. Create a 'data' folder")
        print("  2. Add a test audio file (wav or mp3)")
        print("  3. Update AUDIO_FILE path in the script")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("  • Install required packages: pip install numpy scipy matplotlib")
        print("  • For MP3 support: pip install librosa")
        print("  • Make sure your audio file is not corrupted")
