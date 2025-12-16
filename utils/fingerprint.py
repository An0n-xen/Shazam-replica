import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from config import HashConfig
import hashlib
from utils.visualize import visualize_hash_generation
from utils.audio_utils import create_constellation_map


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

    hash_int = (
        (freq1 & 0x3FF) << 20  # 10 bits for freq1
        | (freq2 & 0x3FF) << 10  # 10 bits for freq2
        | (delta_time & 0x3FF)  # 10 bits for delta_time
    )
    return hash_int


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

    # Entropy check
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


def fingerprint_audio(audio_path, visualize=False, save_plot=None):
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

    # Phase 2: Create constellation map
    peaks, freqs, times, spec = create_constellation_map(
        audio_path, visualize=False  # We'll visualize Phase 3 instead
    )

    # Phase 3: Generate hashes
    hashes = generate_hashes(peaks, times, freqs)

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
