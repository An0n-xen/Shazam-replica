import logging
import hashlib
from math import e
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    # running from main.py
    from utils.config import HashConfig
    from utils.audio_utils import create_constellation_map
    from utils.logging_config import setup_logger
except ImportError:
    # running from this file
    from config import HashConfig
    from audio_utils import create_constellation_map
    from logging_config import setup_logger

# setting up logger
logger = setup_logger(__name__, level=logging.INFO)


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
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 3: Generating Combinatorial Hashes")
    logger.info(f"{'='*60}\n")

    # Sort peaks by time (required for target zone logic)
    peaks_sorted = sorted(peaks, key=lambda x: x[0])

    logger.info(f"Processing {len(peaks_sorted)} peaks...")
    logger.info(f"Fan-out factor: {HashConfig.FAN_OUT}")
    logger.info(
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

    logger.info(f"✓ Generated {len(hashes)} hashes")
    logger.info(f"  Hashes per peak: {len(hashes)/len(peaks_sorted):.1f}")
    logger.info(f"  Expected: ~{HashConfig.FAN_OUT}")

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
    logger.info(f"\n{'='*60}")
    logger.info(f"Hash Distribution Analysis")
    logger.info(f"{'='*60}\n")

    # Count duplicate hashes
    hash_counts = defaultdict(int)
    for hash_val, _ in hashes:
        hash_counts[hash_val] += 1

    unique_hashes = len(hash_counts)
    total_hashes = len(hashes)

    logger.info(f"Total hashes:    {total_hashes}")
    logger.info(f"Unique hashes:   {unique_hashes}")
    logger.info(f"Uniqueness:      {unique_hashes/total_hashes*100:.1f}%")

    # Check for collisions
    duplicates = {h: c for h, c in hash_counts.items() if c > 1}
    logger.info(f"Collisions:      {len(duplicates)}")

    if duplicates:
        max_collision = max(duplicates.values())
        logger.info(
            f"Max collision:   {max_collision} (same hash appears {max_collision} times)"
        )

    # Time distribution
    time_offsets = [t for _, t in hashes]
    logger.info(f"\nTime coverage:")
    logger.info(f"  Min time:      {min(time_offsets):.2f}s")
    logger.info(f"  Max time:      {max(time_offsets):.2f}s")
    logger.info(f"  Duration:      {max(time_offsets) - min(time_offsets):.2f}s")

    # Entropy check
    entropy_score = unique_hashes / total_hashes
    logger.info(f"\nEntropy assessment:")
    if entropy_score > 0.95:
        logger.info(f"  ✓ EXCELLENT ({entropy_score*100:.1f}% unique)")
        logger.info(f"    High specificity, low false positive risk")
    elif entropy_score > 0.85:
        logger.info(f"  ✓ GOOD ({entropy_score*100:.1f}% unique)")
        logger.info(f"    Decent specificity")
    elif entropy_score > 0.70:
        logger.info(f"  ⚠ MODERATE ({entropy_score*100:.1f}% unique)")
        logger.info(f"    Consider increasing fan-out or target zone")
    else:
        logger.info(f"  ✗ LOW ({entropy_score*100:.1f}% unique)")
        logger.info(f"    Too many collisions, adjust parameters")

    logger.info(f"{'='*60}\n")


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
        try:
            from utils.visualize import visualize_hash_generation
        except ImportError:
            from visualize import visualize_hash_generation

        logger.info("\n[Visualizing] Generating hash pair visualization...")
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

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Audio fingerprinting complete!")
    logger.info(f"  File: {audio_path}")
    logger.info(f"  Duration: {metadata['duration']:.2f}s")
    logger.info(f"  Peaks: {metadata['num_peaks']}")
    logger.info(f"  Hashes: {metadata['num_hashes']}")
    logger.info(f"  Density: {metadata['hashes_per_second']:.1f} hashes/second")
    logger.info(f"{'='*60}\n")

    return hashes, metadata


if __name__ == "__main__":
    """
    Test hash generation on a sample audio file.
    """

    AUDIO_FILE = "./data/db_tracks/sample1.mp3"
    SAVE_PLOT = "hash_generation.png"

    hashes, metadata = fingerprint_audio(
        AUDIO_FILE, visualize=True, save_plot=SAVE_PLOT
    )
