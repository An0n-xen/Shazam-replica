import os
import time
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from config import MatchConfig
from database import FingerprintDatabase
from logging_config import setup_logger

# setting up logger
logger = setup_logger(__name__, logging.INFO)


def calculate_time_offsets(time_pairs):
    """
    Calculate time offsets (delta_t) for all matching hash pairs.

    The key insight: If sample matches database, then:
        db_time = sample_time + offset (constant)

    Args:
        time_pairs: List of (sample_time, db_time) tuples

    Returns:
        offsets: List of offset values (db_time - sample_time)
    """
    offsets = []
    for sample_time, db_time in time_pairs:
        offset = db_time - sample_time
        offsets.append(offset)
    return offsets


def find_peak_offset(offsets, bin_size=None):
    """
    Find the most common offset using histogram analysis.
    This detects the "diagonal line" in the scatterplot.

    Args:
        offsets: List of time offset values
        bin_size: Bin size for histogram (seconds)

    Returns:
        peak_offset: Most common offset value
        peak_count: Number of matches at this offset
        histogram: Counter object with all bins
    """
    if not offsets:
        return None, 0, Counter()

    bin_size = bin_size or MatchConfig.BIN_SIZE

    # Bin the offsets
    # Round each offset to nearest bin
    binned_offsets = [round(offset / bin_size) * bin_size for offset in offsets]

    # Count occurrences
    histogram = Counter(binned_offsets)

    # Find peak
    peak_offset, peak_count = histogram.most_common(1)[0]

    return peak_offset, peak_count, histogram


def score_match(time_pairs, track_duration=None, sample_duration=None):
    """
    Score a potential match based on time alignment.

    Args:
        time_pairs: List of (sample_time, db_time) tuples
        track_duration: Duration of database track (optional, for validation)
        sample_duration: Duration of query sample (optional, for validation)

    Returns:
        score: Match score (number of aligned hashes)
        offset: Time offset where match occurs
        confidence: Confidence level (LOW, MEDIUM, HIGH, VERY HIGH)
    """
    if not time_pairs:
        return 0, None, "NONE"

    # Calculate offsets
    offsets = calculate_time_offsets(time_pairs)

    # Optional: Filter invalid offsets
    if track_duration and sample_duration:
        valid_offsets = []
        for offset in offsets:
            # Offset must be: 0 <= offset <= track_duration - sample_duration
            if 0 <= offset <= track_duration - sample_duration:
                valid_offsets.append(offset)
        offsets = valid_offsets

    if not offsets:
        return 0, None, "NONE"

    # Find peak in histogram
    peak_offset, peak_count, histogram = find_peak_offset(offsets)

    # Score is the number of matches at peak offset
    score = peak_count

    # Determine confidence level
    if score >= 50:
        confidence = "VERY HIGH"
    elif score >= 20:
        confidence = "HIGH"
    elif score >= 10:
        confidence = "MEDIUM"
    elif score >= MatchConfig.MIN_MATCHES:
        confidence = "LOW"
    else:
        confidence = "NONE"

    return score, peak_offset, confidence


def match_query(query_hashes, database, return_top_n=5, visualize=False):
    """
    Match a query against the database.

    This is the main search function that identifies which song
    a query sample came from.

    Args:
        query_hashes: List of (hash, sample_time) tuples from query
        database: FingerprintDatabase instance
        return_top_n: Number of top matches to return
        visualize: Whether to show match visualization

    Returns:
        matches: List of match results, sorted by score (best first)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"MATCHING QUERY")
    logger.info(f"{'='*60}\n")

    start_time = time.time()

    # Step 1: Query database for all matching hashes
    logger.info(f"[1/3] Querying database...")
    logger.info(f"  Query hashes: {len(query_hashes)}")

    matches_by_song = database.query_hashes(query_hashes)

    logger.info(f"  Found potential matches in {len(matches_by_song)} song(s)")

    # Step 2: Score each candidate song
    logger.info(f"\n[2/3] Scoring candidates...")

    results = []

    for song_id, time_pairs in matches_by_song.items():
        song_info = database.get_song_info(song_id)

        # Calculate score
        score, offset, confidence = score_match(
            time_pairs,
            track_duration=song_info.get("duration"),
            sample_duration=None,  # We don't know sample duration from hashes alone
        )

        # Only keep if meets minimum threshold
        if score >= MatchConfig.MIN_MATCHES:
            results.append(
                {
                    "song_id": song_id,
                    "score": score,
                    "offset": offset,
                    "confidence": confidence,
                    "num_matching_hashes": len(time_pairs),
                    "song_info": song_info,
                    "time_pairs": time_pairs,  # Keep for visualization
                }
            )

            print(
                f"  Song #{song_id} '{song_info['title'][:30]}': "
                f"score={score}, confidence={confidence}"
            )

    # Step 3: Sort by score (best first)
    results.sort(key=lambda x: x["score"], reverse=True)

    query_time = time.time() - start_time

    logger.info(f"\n[3/3] Results:")
    logger.info(f"  Matches found: {len(results)}")
    logger.info(f"  Query time: {query_time*1000:.1f} ms")

    if results:
        best = results[0]
        logger.info(f"\n✓ BEST MATCH:")
        logger.info(f"  Song: {best['song_info']['title']}")
        logger.info(f"  Artist: {best['song_info']['artist']}")
        logger.info(f"  Score: {best['score']}")
        logger.info(f"  Confidence: {best['confidence']}")
        logger.info(f"  Match at: {best['offset']:.2f}s into the track")

        # Visualize if requested
        if visualize and len(results) > 0:
            from visualize import visualize_match

            visualize_match(results[0], query_hashes)
    else:
        logger.error(f"\n✗ NO MATCH FOUND")
        logger.error(f"  Try adjusting thresholds or adding more songs to database")

    logger.info(f"{'='*60}\n")

    return results[:return_top_n]


def identify_song(audio_path, database, visualize=True):
    """
    Complete identification pipeline:
    Load audio → Fingerprint → Match → Return result

    This is the end-to-end function you'd use in production.

    Args:
        audio_path: Path to query audio file
        database: FingerprintDatabase instance
        visualize: Whether to show match visualization

    Returns:
        best_match: Dict with match info, or None if no match
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"IDENTIFYING SONG")
    logger.info(f"{'='*70}")
    logger.info(f"Query file: {audio_path}\n")

    try:
        from fingerprint import fingerprint_audio

        # Step 1: Fingerprint the query
        logger.info("[Step 1/2] Fingerprinting query audio...")
        query_hashes, metadata = fingerprint_audio(audio_path, visualize=False)

        logger.info(f"  Generated {len(query_hashes)} hashes from query")

        # Step 2: Match against database
        logger.info(f"\n[Step 2/2] Searching database...")
        matches = match_query(
            query_hashes, database, return_top_n=5, visualize=visualize
        )

        if matches:
            return matches[0]  # Return best match
        else:
            return None

    except Exception as e:
        print(f"\n✗ Error during identification: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    """
    Example usage of the matching system.
    """

    db = FingerprintDatabase.load()
    db.print_stats()

    QUERY_FILE = "./data/queries/5s/sample6-5s.mp3"

    if os.path.exists(QUERY_FILE):
        # Identify the song
        match = identify_song(QUERY_FILE, db, visualize=True)
