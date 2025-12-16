import numpy as np
from collections import defaultdict
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, iterate_structure
import librosa
import sounddevice as sd
from scipy.io.wavfile import write


def generate_constellation_map(audio_path):
    print(f"Processing {audio_path}...")
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Spectrogram
    D = librosa.stft(y, n_fft=4096, hop_length=1024)
    spectrogram = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Find Peaks
    neighborhood_size = 20
    local_max = maximum_filter(spectrogram, size=neighborhood_size)
    binary_local_max = spectrogram == local_max

    # Threshold (Note the -40dB fix!)
    background_threshold = -40
    detected_peaks = binary_local_max & (spectrogram > background_threshold)

    # Extract indices
    freq_idx, time_idx = np.where(detected_peaks)

    # Zip and Sort (Crucial Step for Hashing)
    peaks = sorted(zip(time_idx, freq_idx))

    print(f"Found {len(peaks)} peaks.")
    return peaks, (spectrogram, sr), (time_idx, freq_idx)


def create_hashes(peaks):
    hashes = []
    target_zone_size = 15
    fan_out = 10

    for i in range(len(peaks)):
        anchor_t, anchor_f = peaks[i]  # This line was failing before

        for j in range(i + 1, len(peaks)):
            target_t, target_f = peaks[j]

            time_delta = target_t - anchor_t

            if time_delta > target_zone_size:
                break

            if time_delta <= 0:
                continue

            # Create the Fingerprint
            hash_token = f"{anchor_f}|{target_f}|{time_delta}"
            hashes.append((hash_token, anchor_t))

            if (j - i) >= fan_out:
                break

    return hashes


class SimpleShazam:
    def __init__(self):
        # The "Database"
        # Key: Hash Token (e.g., "39|16|2")
        # Value: List of (Song_Name, Absolute_Time_in_Song)
        self.database = defaultdict(list)

    def add_song(self, song_name, fingerprints):
        """Phase 4: Storage - Index a song into the database."""
        print(f"Indexing {song_name} with {len(fingerprints)} hashes...")
        for hash_token, anchor_time in fingerprints:
            self.database[hash_token].append((song_name, anchor_time))

    def match(self, sample_fingerprints):
        """Phase 5: Search & Scoring"""

        # 1. Find Matches
        # We look for each hash from the sample in our big database
        matches = []

        for hash_token, sample_time in sample_fingerprints:
            if hash_token in self.database:
                # We found a match! Retrieve all locations in the DB
                db_hits = self.database[hash_token]

                for song_name, db_time in db_hits:
                    # 2. Calculate Offset (The "Diagonal Line" Logic)
                    # Offset = Database_Time - Sample_Time
                    offset = db_time - sample_time
                    matches.append((song_name, offset))

        # 3. Score via Histogram
        # We group matches by Song, then by Offset
        song_scores = defaultdict(lambda: defaultdict(int))

        for song_name, offset in matches:
            song_scores[song_name][offset] += 1

        # 4. Find the Best Match
        best_song = None
        best_score = 0

        for song_name, offsets in song_scores.items():
            # Find the offset with the most hits (the "Peak" in the histogram)
            # This is the height of the spike in Figure 3B of the paper
            max_hits_at_one_offset = max(offsets.values())

            if max_hits_at_one_offset > best_score:
                best_score = max_hits_at_one_offset
                best_song = song_name
        return best_song, best_score


def record_audio(duration=10, fs=22050):
    print(f"🎤 Listening for {duration} seconds...")

    # Record audio (mono)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    print("✅ Recording complete.")

    # Save to a temporary file (so librosa can load it easily)
    filename = "temp_query.wav"
    # sounddevice returns float32, convert to int16 for wav writing if needed,
    # but scipy often handles floats fine. Let's stick to standard float save.
    write(filename, fs, recording)

    return filename
