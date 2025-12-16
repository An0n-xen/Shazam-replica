"""
Phase 3: Combinatorial Hashing
This creates hash fingerprints from pairs of constellation points.
This is the core of the Shazam algorithm's speed and robustness.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import hashlib

# Import from Phase 2
# (In practice, you'd import these functions from phase2_constellation.py)
# For this demo, we'll assume these are available


# ==================== CONFIGURATION ====================
class HashConfig:
    """Configuration for hash generation"""

    # Target zone parameters (from paper)
    TARGET_T_MIN = 0.0  # Min time ahead (seconds)
    TARGET_T_MAX = 2.0  # Max time ahead (seconds)
    TARGET_F_MIN = 0  # Min frequency difference (bins) - 0 = full spectrum
    TARGET_F_MAX = 1000  # Max frequency difference (bins) - 1000 = essentially full

    # Fan-out: max number of target points per anchor
    FAN_OUT = 10  # Paper suggests 5-15 is good

    # Hash packing
    FREQ_BITS = 10  # Bits for frequency (1024 bins max)
    TIME_BITS = 10  # Bits for time delta (1024 time units max)


# ==================== HASH GENERATION ====================
def generate_hashes(peaks, times, freqs):
    """
    Generate combinatorial hashes from constellation peaks.

    This is the core algorithm: For each anchor point, pair it with
    points in its "target zone" to create unique hash fingerprints.

    Args:
        peaks: List of (time_idx, freq_idx) tuples
        times: Time bins from spectrogram
        freqs: Frequency bins from spectrogram

    Returns:
        hashes: List of (hash, time_offset) tuples
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: Generating Combinatorial Hashes")
    print(f"{'='*60}\n")

    # Sort peaks by time (required for target zone logic)
    peaks_sorted = sorted(peaks, key=lambda x: x[0])

    print(f"Processing {len(peaks_sorted)} peaks...")
    print(f"Fan-out factor: {HashConfig.FAN_OUT}")
    print(
        f"Target zone: {HashConfig.TARGET_T_MIN}s to {HashConfig.TARGET_T_MAX}s ahead\n"
    )

    hashes = []

    # Calculate time resolution (seconds per bin)
    time_resolution = times[1] - times[0] if len(times) > 1 else 1.0

    # Convert target zone time to bins
    target_t_min_bins = int(HashConfig.TARGET_T_MIN / time_resolution)
    target_t_max_bins = int(HashConfig.TARGET_T_MAX / time_resolution)

    # For each peak (anchor point)
    for i, (anchor_t, anchor_f) in enumerate(peaks_sorted):
        # Find target points within the target zone
        targets = []

        # Look ahead in time
        for j in range(i + 1, len(peaks_sorted)):
            target_t, target_f = peaks_sorted[j]

            # Calculate time difference
            delta_t = target_t - anchor_t

            # Check if target is in valid time range
            if delta_t < target_t_min_bins:
                continue  # Too close
            if delta_t > target_t_max_bins:
                break  # Too far (and all subsequent will be too far)

            # Check frequency range (optional - paper uses full spectrum)
            delta_f = abs(target_f - anchor_f)
            if HashConfig.TARGET_F_MIN <= delta_f <= HashConfig.TARGET_F_MAX:
                targets.append((j, target_t, target_f, delta_t))

        # Limit to fan-out factor
        if len(targets) > HashConfig.FAN_OUT:
            # Keep closest targets by time
            targets = sorted(targets, key=lambda x: x[3])[: HashConfig.FAN_OUT]

        # Create hash for each (anchor, target) pair
        for _, target_t, target_f, delta_t in targets:
            # Create hash from (anchor_freq, target_freq, time_delta)
            hash_value = create_hash(anchor_f, target_f, delta_t)

            # Store hash with absolute time of anchor point
            time_offset = times[anchor_t]  # Convert to seconds
            hashes.append((hash_value, time_offset))

    print(f"✓ Generated {len(hashes)} hashes")
    print(f"  Hashes per peak: {len(hashes)/len(peaks_sorted):.1f}")
    print(f"  Expected: ~{HashConfig.FAN_OUT}")

    return hashes


def create_hash(freq1, freq2, delta_time):
    """
    Create a hash from frequency pair and time delta.

    Three approaches are implemented:
    1. Simple tuple (easy to debug)
    2. String (human-readable)
    3. Packed integer (paper's approach, most efficient)

    Args:
        freq1: Anchor frequency bin
        freq2: Target frequency bin
        delta_time: Time difference in bins

    Returns:
        hash_value: Hash as string or integer
    """
    # APPROACH 1: Simple tuple (recommended for debugging)
    # return (freq1, freq2, delta_time)

    # APPROACH 2: String (human-readable, good for testing)
    return f"{freq1:04d}|{freq2:04d}|{delta_time:04d}"

    # APPROACH 3: Packed 32-bit integer (paper's approach, most efficient)
    # Uncomment this for production use:
    """
    # Pack into 32 bits: 10 bits freq1, 10 bits freq2, 10 bits delta_t
    hash_int = (
        (freq1 & 0x3FF) << 20 |  # 10 bits for freq1
        (freq2 & 0x3FF) << 10 |  # 10 bits for freq2
        (delta_time & 0x3FF)     # 10 bits for delta_time
    )
    return hash_int
    """


# ==================== HASH VISUALIZATION ====================
def visualize_hash_generation(
    peaks, times, freqs, hashes, num_examples=5, save_path=None
):
    """
    Visualize how hashes are created from peak pairs.
    Shows anchor points and their target zones.

    Args:
        peaks: List of (time_idx, freq_idx) tuples
        times: Time bins
        freqs: Frequency bins
        hashes: Generated hashes
        num_examples: Number of anchor points to highlight
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all peaks as gray dots
    peak_times = [times[t_idx] for t_idx, f_idx in peaks]
    peak_freqs = [freqs[f_idx] for t_idx, f_idx in peaks]
    ax.scatter(peak_times, peak_freqs, c="gray", s=20, alpha=0.5, label="All peaks")

    # Sort peaks by time
    peaks_sorted = sorted(peaks, key=lambda x: x[0])

    # Time resolution
    time_resolution = times[1] - times[0]

    # Highlight a few anchor points and their target zones
    colors = plt.cm.tab10(np.linspace(0, 1, num_examples))

    for idx, color in zip(
        range(0, len(peaks_sorted), max(1, len(peaks_sorted) // num_examples)), colors
    ):
        if idx >= len(peaks_sorted):
            break

        anchor_t_idx, anchor_f_idx = peaks_sorted[idx]
        anchor_t = times[anchor_t_idx]
        anchor_f = freqs[anchor_f_idx]

        # Plot anchor point
        ax.scatter(
            [anchor_t],
            [anchor_f],
            c=[color],
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            label=f"Anchor {idx+1}",
            zorder=5,
        )

        # Draw target zone
        zone_t_min = anchor_t + HashConfig.TARGET_T_MIN
        zone_t_max = anchor_t + HashConfig.TARGET_T_MAX
        zone_f_min = freqs[max(0, anchor_f_idx - HashConfig.TARGET_F_MAX)]
        zone_f_max = freqs[min(len(freqs) - 1, anchor_f_idx + HashConfig.TARGET_F_MAX)]

        # Draw rectangle for target zone
        from matplotlib.patches import Rectangle

        rect = Rectangle(
            (zone_t_min, zone_f_min),
            zone_t_max - zone_t_min,
            zone_f_max - zone_f_min,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Find and highlight target points for this anchor
        target_t_min_bins = int(HashConfig.TARGET_T_MIN / time_resolution)
        target_t_max_bins = int(HashConfig.TARGET_T_MAX / time_resolution)

        targets = []
        for j in range(idx + 1, len(peaks_sorted)):
            target_t_idx, target_f_idx = peaks_sorted[j]
            delta_t = target_t_idx - anchor_t_idx

            if delta_t < target_t_min_bins:
                continue
            if delta_t > target_t_max_bins:
                break

            targets.append((target_t_idx, target_f_idx))

            if len(targets) >= HashConfig.FAN_OUT:
                break

        # Draw lines from anchor to targets
        for target_t_idx, target_f_idx in targets:
            target_t = times[target_t_idx]
            target_f = freqs[target_f_idx]
            ax.plot(
                [anchor_t, target_t],
                [anchor_f, target_f],
                color=color,
                alpha=0.3,
                linewidth=1,
                zorder=3,
            )

        # Plot target points
        if targets:
            target_times = [times[t_idx] for t_idx, _ in targets]
            target_freqs = [freqs[f_idx] for _, f_idx in targets]
            ax.scatter(
                target_times,
                target_freqs,
                c=[color],
                s=100,
                marker="o",
                edgecolors="black",
                linewidths=1,
                alpha=0.7,
                zorder=4,
            )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.set_title(
        "Combinatorial Hash Generation\n"
        + f"(Stars = Anchors, Lines = Hash Pairs, Dashed boxes = Target Zones)",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Visualization saved to: {save_path}")

    plt.show()


def print_hash_examples(hashes, num_examples=10):
    """
    Print some example hashes to verify they look reasonable.

    Args:
        hashes: List of (hash, time_offset) tuples
        num_examples: Number to print
    """
    print(f"\n{'='*60}")
    print(f"Example Hashes (first {num_examples}):")
    print(f"{'='*60}")
    print(f"{'Hash':<30} {'Time Offset (s)':<15}")
    print(f"{'-'*60}")

    for i, (hash_val, time_offset) in enumerate(hashes[:num_examples]):
        print(f"{str(hash_val):<30} {time_offset:>10.3f}")

    print(f"{'-'*60}")
    print(f"Total hashes: {len(hashes)}")
    print(f"{'='*60}\n")


def analyze_hash_distribution(hashes):
    """
    Analyze the hash distribution to check for good entropy.

    Args:
        hashes: List of (hash, time_offset) tuples
    """
    print(f"\n{'='*60}")
    print(f"Hash Distribution Analysis")
    print(f"{'='*60}\n")

    # Count duplicate hashes
    hash_counts = defaultdict(int)
    for hash_val, _ in hashes:
        hash_counts[hash_val] += 1

    unique_hashes = len(hash_counts)
    total_hashes = len(hashes)

    print(f"Total hashes:    {total_hashes}")
    print(f"Unique hashes:   {unique_hashes}")
    print(f"Uniqueness:      {unique_hashes/total_hashes*100:.1f}%")

    # Check for collisions
    duplicates = {h: c for h, c in hash_counts.items() if c > 1}
    print(f"Collisions:      {len(duplicates)}")

    if duplicates:
        max_collision = max(duplicates.values())
        print(
            f"Max collision:   {max_collision} (same hash appears {max_collision} times)"
        )

    # Time distribution
    time_offsets = [t for _, t in hashes]
    print(f"\nTime coverage:")
    print(f"  Min time:      {min(time_offsets):.2f}s")
    print(f"  Max time:      {max(time_offsets):.2f}s")
    print(f"  Duration:      {max(time_offsets) - min(time_offsets):.2f}s")

    # Entropy check (good hashes should have high uniqueness)
    entropy_score = unique_hashes / total_hashes
    print(f"\nEntropy assessment:")
    if entropy_score > 0.95:
        print(f"  ✓ EXCELLENT ({entropy_score*100:.1f}% unique)")
        print(f"    High specificity, low false positive risk")
    elif entropy_score > 0.85:
        print(f"  ✓ GOOD ({entropy_score*100:.1f}% unique)")
        print(f"    Decent specificity")
    elif entropy_score > 0.70:
        print(f"  ⚠ MODERATE ({entropy_score*100:.1f}% unique)")
        print(f"    Consider increasing fan-out or target zone")
    else:
        print(f"  ✗ LOW ({entropy_score*100:.1f}% unique)")
        print(f"    Too many collisions, adjust parameters")

    print(f"{'='*60}\n")


# ==================== INTEGRATION WITH PHASE 2 ====================
def fingerprint_audio(audio_path, visualize=True, save_plot=None):
    """
    Complete pipeline: Phase 2 + Phase 3
    Load audio → Create constellation → Generate hashes

    This is what you'd call to fingerprint a song for the database.

    Args:
        audio_path: Path to audio file
        visualize: Whether to show plots
        save_plot: Path to save visualization

    Returns:
        hashes: List of (hash, time_offset) tuples
        metadata: Dict with additional info
    """
    # Import Phase 2 functions (in practice, import from phase2_constellation.py)
    # For this demo, assuming they're available
    try:
        from phase2_constellation import create_constellation_map
    except ImportError:
        print(
            "⚠ Warning: Could not import Phase 2. Make sure phase2_constellation.py is available."
        )
        print("For now, using mock data...")
        # Create mock data for demonstration
        peaks = [(i * 10, i * 5) for i in range(50)]
        times = np.linspace(0, 30, 300)
        freqs = np.linspace(0, 11025, 2049)
        return None, None

    # Phase 2: Create constellation map
    peaks, freqs, times, spec = create_constellation_map(
        audio_path, visualize=False  # We'll visualize Phase 3 instead
    )

    # Phase 3: Generate hashes
    hashes = generate_hashes(peaks, times, freqs)

    # Print examples
    print_hash_examples(hashes)

    # Analyze distribution
    analyze_hash_distribution(hashes)

    # Visualize (optional)
    if visualize:
        print("\n[Visualizing] Generating hash pair visualization...")
        visualize_hash_generation(
            peaks, times, freqs, hashes, num_examples=5, save_path=save_plot
        )

    # Prepare metadata
    metadata = {
        "num_peaks": len(peaks),
        "num_hashes": len(hashes),
        "duration": times[-1],
        "hashes_per_second": len(hashes) / times[-1],
    }

    print(f"\n{'='*60}")
    print(f"✓ Audio fingerprinting complete!")
    print(f"  File: {audio_path}")
    print(f"  Duration: {metadata['duration']:.2f}s")
    print(f"  Peaks: {metadata['num_peaks']}")
    print(f"  Hashes: {metadata['num_hashes']}")
    print(f"  Density: {metadata['hashes_per_second']:.1f} hashes/second")
    print(f"{'='*60}\n")

    return hashes, metadata


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    """
    Test hash generation on a sample audio file.

    Usage:
        python phase3_hashing.py

    Requirements:
        - phase2_constellation.py must be in the same directory
        - An audio file to test with
    """

    # === CONFIGURATION ===
    AUDIO_FILE = "./data/db_tracks/sample1.mp3"  # Change to your file
    SAVE_PLOT = "hash_generation.png"

    print(
        """
    ╔════════════════════════════════════════════════════════════╗
    ║         Phase 3: Combinatorial Hash Generation              ║
    ╚════════════════════════════════════════════════════════════╝
    
    This script will:
    1. Load audio and create constellation map (Phase 2)
    2. Generate combinatorial hashes (Phase 3)
    3. Analyze hash quality
    4. Visualize the hash pair creation
    
    Each hash represents a unique pair of time-frequency points.
    """
    )

    try:
        # Run complete fingerprinting
        hashes, metadata = fingerprint_audio(
            AUDIO_FILE, visualize=True, save_plot=SAVE_PLOT
        )

        print("\n✅ SUCCESS! Hash generation complete.")
        print("\nWhat to look for:")
        print("  • High uniqueness (>95% is excellent)")
        print("  • Few collisions (duplicate hashes)")
        print("  • ~50-150 hashes per second of audio")
        print("  • Visualization shows anchor stars connected to targets")

        print("\nNext steps:")
        print("  • Adjust HashConfig if needed")
        print("  • Test with multiple songs")
        print("  • Move to Phase 4: Database Storage")

    except FileNotFoundError:
        print(f"\n❌ ERROR: Could not find audio file: {AUDIO_FILE}")
        print("\nMake sure:")
        print("  1. phase2_constellation.py is in the same directory")
        print("  2. You have an audio file ready")
        print("  3. Update AUDIO_FILE path in the script")

    except ImportError as e:
        print(f"\n❌ ERROR: Missing dependency: {e}")
        print("\nInstall required packages:")
        print("  pip install numpy scipy matplotlib librosa")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
