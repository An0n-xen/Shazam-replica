"""
Phase 5: Search & Matching Algorithm
This implements the "diagonal line" detection to identify songs.
The core insight: matching hashes should have consistent time offsets.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import time
from pathlib import Path


# ==================== CONFIGURATION ====================
class MatchConfig:
    """Configuration for matching algorithm"""

    # Matching thresholds
    MIN_MATCHES = 5  # Minimum number of aligned hashes for a match

    # Histogram binning
    BIN_SIZE = 0.5  # Seconds - cluster offsets within this window

    # Confidence scoring
    CONFIDENCE_THRESHOLD = 5  # Min score for confident match


# ==================== TIME OFFSET ANALYSIS ====================
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


# ==================== MATCHING FUNCTION ====================
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
    print(f"\n{'='*60}")
    print(f"MATCHING QUERY")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Step 1: Query database for all matching hashes
    print(f"[1/3] Querying database...")
    print(f"  Query hashes: {len(query_hashes)}")

    matches_by_song = database.query_hashes(query_hashes)

    print(f"  Found potential matches in {len(matches_by_song)} song(s)")

    # Step 2: Score each candidate song
    print(f"\n[2/3] Scoring candidates...")

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

    print(f"\n[3/3] Results:")
    print(f"  Matches found: {len(results)}")
    print(f"  Query time: {query_time*1000:.1f} ms")

    if results:
        best = results[0]
        print(f"\n✓ BEST MATCH:")
        print(f"  Song: {best['song_info']['title']}")
        print(f"  Artist: {best['song_info']['artist']}")
        print(f"  Score: {best['score']}")
        print(f"  Confidence: {best['confidence']}")
        print(f"  Match at: {best['offset']:.2f}s into the track")

        # Visualize if requested
        if visualize and len(results) > 0:
            visualize_match(results[0], query_hashes)
    else:
        print(f"\n✗ NO MATCH FOUND")
        print(f"  Try adjusting thresholds or adding more songs to database")

    print(f"{'='*60}\n")

    return results[:return_top_n]


# ==================== VISUALIZATION ====================
def visualize_match(match_result, query_hashes):
    """
    Visualize the match using scatterplot and histogram.

    Creates two plots:
    1. Scatterplot: db_time vs sample_time (should show diagonal line)
    2. Histogram: distribution of time offsets (should show clear peak)

    Args:
        match_result: Dict with match information
        query_hashes: Original query hashes for context
    """
    time_pairs = match_result["time_pairs"]
    song_info = match_result["song_info"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Scatterplot (Database Time vs Sample Time)
    sample_times = [s_time for s_time, _ in time_pairs]
    db_times = [db_time for _, db_time in time_pairs]

    ax1.scatter(sample_times, db_times, alpha=0.6, s=30, c="steelblue")

    # Draw ideal diagonal line (if we know the offset)
    if match_result["offset"] is not None:
        offset = match_result["offset"]
        x_range = [0, max(sample_times) if sample_times else 10]
        y_range = [offset, offset + x_range[1]]
        ax1.plot(
            x_range,
            y_range,
            "r--",
            linewidth=2,
            label=f"Match line (offset={offset:.1f}s)",
            alpha=0.7,
        )

    ax1.set_xlabel("Sample Time (s)", fontsize=11)
    ax1.set_ylabel("Database Time (s)", fontsize=11)
    ax1.set_title(
        f"Time Alignment Scatterplot\n" f'Song: {song_info["title"][:40]}', fontsize=12
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of Time Offsets
    offsets = calculate_time_offsets(time_pairs)

    ax2.hist(offsets, bins=30, color="steelblue", alpha=0.7, edgecolor="black")

    # Mark the peak
    peak_offset = match_result["offset"]
    if peak_offset is not None:
        ax2.axvline(
            peak_offset,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Peak at {peak_offset:.2f}s",
        )

    ax2.set_xlabel("Time Offset (db_time - sample_time) [s]", fontsize=11)
    ax2.set_ylabel("Number of Matches", fontsize=11)
    ax2.set_title(
        f"Offset Histogram\n"
        f'Score: {match_result["score"]}, '
        f'Confidence: {match_result["confidence"]}',
        fontsize=12,
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("match_visualization.png", dpi=150, bbox_inches="tight")
    print(f"  Visualization saved: match_visualization.png")
    plt.show()


def visualize_all_candidates(matches_by_song, database, query_hashes, top_n=4):
    """
    Visualize multiple candidate matches for comparison.
    Shows why the top match is better than others.

    Args:
        matches_by_song: Dict of song_id → time_pairs
        database: FingerprintDatabase
        query_hashes: Query hashes
        top_n: Number of candidates to show
    """
    # Score all candidates
    candidates = []
    for song_id, time_pairs in matches_by_song.items():
        song_info = database.get_song_info(song_id)
        score, offset, confidence = score_match(time_pairs)
        candidates.append(
            {
                "song_id": song_id,
                "score": score,
                "offset": offset,
                "time_pairs": time_pairs,
                "song_info": song_info,
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:top_n]

    # Create subplots
    n = len(candidates)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, candidate in enumerate(candidates):
        ax = axes[idx]

        # Scatterplot
        sample_times = [s for s, _ in candidate["time_pairs"]]
        db_times = [d for _, d in candidate["time_pairs"]]

        ax.scatter(sample_times, db_times, alpha=0.6, s=20)

        # Diagonal line
        if candidate["offset"] is not None:
            x_range = [0, max(sample_times) if sample_times else 10]
            y_range = [candidate["offset"], candidate["offset"] + x_range[1]]
            ax.plot(x_range, y_range, "r--", linewidth=1.5, alpha=0.7)

        # Title
        title = f"#{idx+1}: {candidate['song_info']['title'][:20]}\n"
        title += f"Score: {candidate['score']}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Sample Time (s)", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Database Time (s)", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("all_candidates.png", dpi=150, bbox_inches="tight")
    print(f"  Candidate comparison saved: all_candidates.png")
    plt.show()


# ==================== IDENTIFICATION FUNCTION ====================
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
    print(f"\n{'='*70}")
    print(f"IDENTIFYING SONG")
    print(f"{'='*70}")
    print(f"Query file: {audio_path}\n")

    try:
        # Import fingerprinting from Phase 3
        from phase3_hashing import fingerprint_audio

        # Step 1: Fingerprint the query
        print("[Step 1/2] Fingerprinting query audio...")
        query_hashes, metadata = fingerprint_audio(audio_path, visualize=False)

        print(f"  Generated {len(query_hashes)} hashes from query")

        # Step 2: Match against database
        print(f"\n[Step 2/2] Searching database...")
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


# ==================== BATCH TESTING ====================
def test_recognition_accuracy(test_dir, database):
    """
    Test recognition accuracy on a directory of query clips.

    Useful for measuring performance on your test set.

    Args:
        test_dir: Directory containing test query audio files
        database: FingerprintDatabase instance

    Returns:
        results: Dict with accuracy metrics
    """
    from pathlib import Path

    print(f"\n{'='*60}")
    print(f"BATCH TESTING")
    print(f"{'='*60}\n")

    test_dir = Path(test_dir)
    query_files = list(test_dir.glob("*.wav")) + list(test_dir.glob("*.mp3"))

    if not query_files:
        print(f"No test files found in {test_dir}")
        return None

    print(f"Found {len(query_files)} test files\n")

    total = len(query_files)
    correct = 0
    no_match = 0
    wrong_match = 0

    results = []

    for i, query_file in enumerate(query_files, 1):
        print(f"\n[{i}/{total}] Testing: {query_file.name}")
        print("-" * 60)

        # Identify
        match = identify_song(str(query_file), database, visualize=False)

        # In a real test, you'd compare against ground truth
        # For now, we just record if we got any match
        if match is None:
            no_match += 1
            result = "NO_MATCH"
        elif match["confidence"] in ["HIGH", "VERY HIGH"]:
            correct += 1
            result = "MATCH"
        else:
            result = "LOW_CONFIDENCE"

        results.append({"file": query_file.name, "result": result, "match": match})

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries:        {total}")
    print(f"High confidence:      {correct} ({correct/total*100:.1f}%)")
    print(f"No match:             {no_match} ({no_match/total*100:.1f}%)")
    print(f"{'='*60}\n")

    return results


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    """
    Example usage of the matching system.

    This demonstrates:
    1. Loading a database
    2. Identifying a query song
    3. Visualization of results
    """

    print(
        """
    ╔════════════════════════════════════════════════════════════╗
    ║         Phase 5: Search & Matching Algorithm                ║
    ╚════════════════════════════════════════════════════════════╝
    
    This is the final piece! It:
    1. Takes a query audio clip
    2. Matches it against your database
    3. Returns the song name and timestamp
    """
    )

    # === STEP 1: Load Database ===
    print("\n" + "=" * 60)
    print("STEP 1: Loading Database")
    print("=" * 60)

    try:
        from phase4_database import FingerprintDatabase

        db = FingerprintDatabase.load()
        db.print_stats()

        if len(db.song_metadata) == 0:
            print("\n⚠ Database is empty!")
            print("\nPlease run Phase 4 first to build a database:")
            print("  python phase4_database.py --dir ./data/db_tracks")
            exit(1)

    except FileNotFoundError:
        print("\n✗ No database found!")
        print("\nPlease run Phase 4 first to build a database:")
        print("  python phase4_database.py --dir ./data/db_tracks")
        exit(1)

    # === STEP 2: Identify a Song ===
    print("\n" + "=" * 60)
    print("STEP 2: Identifying Query Song")
    print("=" * 60)

    # CHANGE THIS to your query file!
    QUERY_FILE = "./data/queries/5s/sample3-5s.m4a"

    import os

    if os.path.exists(QUERY_FILE):
        # Identify the song
        match = identify_song(QUERY_FILE, db, visualize=True)

        if match:
            print("\n" + "=" * 60)
            print("✅ IDENTIFICATION SUCCESSFUL!")
            print("=" * 60)
            print(f"Song:       {match['song_info']['title']}")
            print(f"Artist:     {match['song_info']['artist']}")
            print(f"Score:      {match['score']}")
            print(f"Confidence: {match['confidence']}")
            print(f"Position:   {match['offset']:.2f} seconds into track")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ NO MATCH FOUND")
            print("=" * 60)
            print("Possible reasons:")
            print("  • Song not in database")
            print("  • Too much noise in recording")
            print("  • Query too short (try 10+ seconds)")
            print("  • Thresholds too strict")
            print("=" * 60)

    else:
        print(f"\n⚠ Query file not found: {QUERY_FILE}")
        print("\nTo test matching:")
        print("  1. Create a test query:")
        print("     • Crop 10 seconds from one of your database songs")
        print("     • Or record a song playing from speakers")
        print("  2. Save to: data/queries/test_clip.wav")
        print("  3. Update QUERY_FILE in the script")
        print("  4. Run this script again")

        print("\n" + "=" * 60)
        print("DEMO: Simulating a match with database song")
        print("=" * 60)

        # Demo with a song from the database
        if db.song_metadata:
            first_song = db.get_song_info(1)
            print(f"\nUsing database song: {first_song['filename']}")
            print("(In a real scenario, use a separate query clip)")

            # You could test with the actual database file here
            # match = identify_song(first_song['filepath'], db, visualize=True)

    # === STEP 3: Batch Testing (Optional) ===
    print("\n" + "=" * 60)
    print("STEP 3: Batch Testing (Optional)")
    print("=" * 60)

    TEST_DIR = "../data/queries/10s"
    if os.path.exists(TEST_DIR) and os.path.isdir(TEST_DIR):
        test_files = list(Path(TEST_DIR).glob("*.wav")) + list(
            Path(TEST_DIR).glob("*.mp3")
        )
        if test_files:
            print(f"\nFound {len(test_files)} test files")
            response = input("Run batch test? (y/n): ")
            if response.lower() == "y":
                test_recognition_accuracy(TEST_DIR, db)
        else:
            print(f"No test files in {TEST_DIR}")

    print("\n" + "=" * 60)
    print("✅ Phase 5 Complete!")
    print("=" * 60)
    print("\nYou now have a working Shazam-like system!")
    print("\nNext steps:")
    print("  • Test with noisy recordings")
    print("  • Tune matching thresholds")
    print("  • Scale to larger database")
    print("  • Add real-time recording feature")
    print("=" * 60 + "\n")
