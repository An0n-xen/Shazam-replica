import librosa
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from config import HashConfig

from matcher import score_match, calculate_time_offsets


def plot_constellation(audio_path: str, generate_constellation_map: Callable):
    _, graph_things, time_freq = generate_constellation_map(audio_path)
    S, sr = graph_things
    time_idx, freq_idx = time_freq

    plt.figure(figsize=(12, 6))

    # Plot the background Spectrogram
    librosa.display.specshow(
        S, sr=sr, hop_length=1024, x_axis="time", y_axis="hz", cmap="viridis", alpha=0.6
    )

    # Convert indices to units for plotting
    times = librosa.frames_to_time(time_idx, sr=sr, hop_length=1024)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)[freq_idx]

    plt.scatter(times, freqs, color="r", s=5, label="Constellation Peaks")
    plt.title(f"Constellation Map: {audio_path}")
    plt.legend()
    plt.ylim(0, 8000)  # Most music energy is below 8kHz
    plt.show()


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
