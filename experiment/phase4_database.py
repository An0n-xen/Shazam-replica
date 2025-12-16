"""
Phase 4: Database Storage & Indexing
This creates an efficient in-memory database for fast hash lookups.
The database maps: hash → [(time_offset, song_id), ...]
"""

import os
import pickle
import json
from collections import defaultdict
from pathlib import Path
import time


# ==================== CONFIGURATION ====================
class DatabaseConfig:
    """Configuration for database storage"""

    # Storage paths
    DB_FILE = "fingerprint_database.pkl"  # Pickled database
    METADATA_FILE = "song_metadata.json"  # Song information

    # Database format
    # We use a dictionary: hash → [(time_offset, song_id), ...]
    # This allows multiple songs to share the same hash (collision handling)


# ==================== DATABASE CLASS ====================
class FingerprintDatabase:
    """
    In-memory hash database for audio fingerprinting.

    Structure:
        hash_table: {hash: [(time_offset, song_id), ...]}
        song_metadata: {song_id: {filename, title, duration, num_hashes, ...}}
    """

    def __init__(self):
        """Initialize empty database"""
        self.hash_table = defaultdict(list)
        self.song_metadata = {}
        self.next_song_id = 1

    def add_song(self, hashes, song_path, metadata=None):
        """
        Add a song to the database.

        Args:
            hashes: List of (hash, time_offset) tuples from Phase 3
            song_path: Path to the audio file
            metadata: Optional dict with additional info (title, artist, etc.)

        Returns:
            song_id: Integer ID assigned to this song
        """
        song_id = self.next_song_id
        self.next_song_id += 1

        # Add hashes to the hash table
        for hash_val, time_offset in hashes:
            self.hash_table[hash_val].append((time_offset, song_id))

        # Store song metadata
        filename = os.path.basename(song_path)
        self.song_metadata[song_id] = {
            "song_id": song_id,
            "filename": filename,
            "filepath": song_path,
            "num_hashes": len(hashes),
            "title": metadata.get("title", filename) if metadata else filename,
            "artist": metadata.get("artist", "Unknown") if metadata else "Unknown",
            "duration": metadata.get("duration", 0) if metadata else 0,
            "indexed_at": time.time(),
        }

        print(f"✓ Added song #{song_id}: {filename}")
        print(f"  Hashes: {len(hashes)}")

        return song_id

    def get_song_info(self, song_id):
        """Get metadata for a song by ID"""
        return self.song_metadata.get(song_id)

    def get_all_songs(self):
        """Get list of all songs in database"""
        return list(self.song_metadata.values())

    def lookup_hash(self, hash_val):
        """
        Look up a hash in the database.

        Args:
            hash_val: Hash to look up

        Returns:
            List of (time_offset, song_id) tuples, or empty list if not found
        """
        return self.hash_table.get(hash_val, [])

    def query_hashes(self, query_hashes):
        """
        Look up multiple hashes (from a query sample).

        Args:
            query_hashes: List of (hash, sample_time) tuples

        Returns:
            matches: Dict mapping song_id to list of (sample_time, db_time) pairs
        """
        matches = defaultdict(list)

        for hash_val, sample_time in query_hashes:
            # Look up this hash in database
            results = self.lookup_hash(hash_val)

            # Group by song_id
            for db_time, song_id in results:
                matches[song_id].append((sample_time, db_time))

        return matches

    def get_stats(self):
        """Get database statistics"""
        total_hashes = sum(len(v) for v in self.hash_table.values())
        unique_hashes = len(self.hash_table)

        return {
            "num_songs": len(self.song_metadata),
            "unique_hashes": unique_hashes,
            "total_hash_entries": total_hashes,
            "avg_hashes_per_song": total_hashes / max(1, len(self.song_metadata)),
            "avg_collisions": total_hashes / max(1, unique_hashes),
        }

    def save(self, db_path=None, metadata_path=None):
        """
        Save database to disk.

        Args:
            db_path: Path for database file (pickle)
            metadata_path: Path for metadata file (json)
        """
        db_path = db_path or DatabaseConfig.DB_FILE
        metadata_path = metadata_path or DatabaseConfig.METADATA_FILE

        # Save hash table (binary pickle)
        with open(db_path, "wb") as f:
            # Convert defaultdict to regular dict for pickling
            data = {
                "hash_table": dict(self.hash_table),
                "next_song_id": self.next_song_id,
            }
            pickle.dump(data, f)

        # Save metadata (JSON for human readability)
        with open(metadata_path, "w") as f:
            json.dump(self.song_metadata, f, indent=2)

        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"\n✓ Database saved:")
        print(f"  Hash table: {db_path} ({db_size_mb:.2f} MB)")
        print(f"  Metadata: {metadata_path}")

    @classmethod
    def load(cls, db_path=None, metadata_path=None):
        """
        Load database from disk.

        Args:
            db_path: Path to database file
            metadata_path: Path to metadata file

        Returns:
            FingerprintDatabase instance
        """
        db_path = db_path or DatabaseConfig.DB_FILE
        metadata_path = metadata_path or DatabaseConfig.METADATA_FILE

        db = cls()

        # Load hash table
        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                data = pickle.load(f)
                db.hash_table = defaultdict(list, data["hash_table"])
                db.next_song_id = data["next_song_id"]
            print(f"✓ Loaded hash table from {db_path}")
        else:
            print(f"⚠ No database file found at {db_path}")

        # Load metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                # JSON keys are strings, convert back to int
                metadata = json.load(f)
                db.song_metadata = {int(k): v for k, v in metadata.items()}
            print(f"✓ Loaded metadata from {metadata_path}")
        else:
            print(f"⚠ No metadata file found at {metadata_path}")

        return db

    def print_stats(self):
        """Print database statistics"""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"DATABASE STATISTICS")
        print(f"{'='*60}")
        print(f"Songs in database:     {stats['num_songs']}")
        print(f"Unique hashes:         {stats['unique_hashes']:,}")
        print(f"Total hash entries:    {stats['total_hash_entries']:,}")
        print(f"Avg hashes per song:   {stats['avg_hashes_per_song']:.1f}")
        print(f"Avg collision rate:    {stats['avg_collisions']:.2f} songs/hash")
        print(f"{'='*60}\n")

        # List songs
        if self.song_metadata:
            print("Songs in database:")
            print(f"{'ID':<5} {'Title':<30} {'Hashes':<10} {'Duration':<10}")
            print(f"{'-'*60}")
            for song_id, info in sorted(self.song_metadata.items()):
                title = (
                    info["title"][:28] + ".."
                    if len(info["title"]) > 30
                    else info["title"]
                )
                duration = f"{info['duration']:.1f}s" if info["duration"] > 0 else "N/A"
                print(
                    f"{song_id:<5} {title:<30} {info['num_hashes']:<10} {duration:<10}"
                )
            print(f"{'-'*60}\n")


# ==================== INDEXING FUNCTIONS ====================
def index_audio_file(audio_path, database, metadata=None):
    """
    Index a single audio file into the database.

    This runs Phase 2 + Phase 3 + adds to database.

    Args:
        audio_path: Path to audio file
        database: FingerprintDatabase instance
        metadata: Optional dict with title, artist, etc.

    Returns:
        song_id: ID assigned to this song
    """
    print(f"\nIndexing: {audio_path}")
    print(f"{'-'*60}")

    try:
        # Import fingerprinting function from Phase 3
        from phase3_hashing import fingerprint_audio

        # Generate fingerprints (Phase 2 + 3)
        hashes, fp_metadata = fingerprint_audio(
            audio_path, visualize=False  # Don't show plots during batch indexing
        )

        # Combine metadata
        if metadata is None:
            metadata = {}
        metadata["duration"] = fp_metadata["duration"]

        # Add to database
        song_id = database.add_song(hashes, audio_path, metadata)

        return song_id

    except Exception as e:
        print(f"✗ Error indexing {audio_path}: {e}")
        return None


def index_directory(directory_path, database, pattern="*.wav"):
    """
    Index all audio files in a directory.

    Args:
        directory_path: Path to directory containing audio files
        database: FingerprintDatabase instance
        pattern: File pattern to match (e.g., "*.wav", "*.mp3")

    Returns:
        List of song_ids that were indexed
    """
    print(f"\n{'='*60}")
    print(f"INDEXING DIRECTORY: {directory_path}")
    print(f"Pattern: {pattern}")
    print(f"{'='*60}\n")

    # Find all matching audio files
    directory = Path(directory_path)
    audio_files = list(directory.glob(pattern))

    # Also try other common formats
    for ext in ["*.mp3", "*.flac", "*.m4a", "*.ogg"]:
        if ext != pattern:
            audio_files.extend(directory.glob(ext))

    if not audio_files:
        print(f"⚠ No audio files found in {directory_path}")
        return []

    print(f"Found {len(audio_files)} audio files")

    indexed_ids = []
    start_time = time.time()

    # Index each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

        # Extract metadata from filename if possible
        metadata = {
            "title": audio_file.stem,  # Filename without extension
            "artist": "Unknown",
        }

        song_id = index_audio_file(str(audio_file), database, metadata)

        if song_id:
            indexed_ids.append(song_id)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"✓ INDEXING COMPLETE")
    print(f"  Successfully indexed: {len(indexed_ids)}/{len(audio_files)} files")
    print(f"  Time taken: {elapsed:.1f} seconds")
    print(f"  Avg time per file: {elapsed/max(1, len(indexed_ids)):.1f} seconds")
    print(f"{'='*60}\n")

    return indexed_ids


# ==================== COMMAND LINE INTERFACE ====================
def build_database_cli():
    """
    Command-line interface for building the database.

    Usage:
        python phase4_database.py
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build audio fingerprint database (Phase 4)"
    )

    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing audio files to index",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="File pattern to match (default: *.wav)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DatabaseConfig.DB_FILE,
        help="Output database file (default: fingerprint_database.pkl)",
    )

    args = parser.parse_args()

    # Create database
    print("\nCreating new fingerprint database...")
    db = FingerprintDatabase()

    # Index directory
    index_directory(args.dir, db, pattern=args.pattern)

    # Print stats
    db.print_stats()

    # Save database
    db.save(db_path=args.output)

    print("\n✅ Database build complete!")


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    """
    Example usage of the database system.

    This shows how to:
    1. Create a database
    2. Index audio files
    3. Save/load database
    4. Query the database
    """

    print(
        """
    ╔════════════════════════════════════════════════════════════╗
    ║           Phase 4: Database Storage & Indexing              ║
    ╚════════════════════════════════════════════════════════════╝
    
    This script demonstrates:
    1. Creating an in-memory hash database
    2. Indexing audio files (running Phase 2 + 3)
    3. Saving/loading the database
    4. Querying for matches
    """
    )

    # === EXAMPLE 1: Create and populate database ===
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Building Database")
    print("=" * 60)

    # Create new database
    db = FingerprintDatabase()

    # Index a directory of songs
    # CHANGE THIS to your audio directory!
    AUDIO_DIR = "data/db_tracks"

    if os.path.exists(AUDIO_DIR):
        indexed = index_directory(AUDIO_DIR, db)

        # Show statistics
        db.print_stats()

        # Save to disk
        db.save()

    else:
        print(f"\n⚠ Directory not found: {AUDIO_DIR}")
        print("\nTo use this script:")
        print("  1. Create 'data/db_tracks/' directory")
        print("  2. Add 5-10 audio files (wav or mp3)")
        print("  3. Run this script again")
        print("\nOr use the CLI:")
        print("  python phase4_database.py --dir ./your_music_folder")

    # === EXAMPLE 2: Load existing database ===
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Loading Existing Database")
    print("=" * 60)

    if os.path.exists(DatabaseConfig.DB_FILE):
        # Load from disk
        db_loaded = FingerprintDatabase.load()
        db_loaded.print_stats()

        # === EXAMPLE 3: Query demonstration ===
        print("\n" + "=" * 60)
        print("EXAMPLE 3: Hash Lookup Demo")
        print("=" * 60)

        # Get a sample hash from the database
        if db_loaded.hash_table:
            sample_hash = next(iter(db_loaded.hash_table.keys()))
            results = db_loaded.lookup_hash(sample_hash)

            print(f"\nExample hash lookup:")
            print(f"  Hash: {sample_hash}")
            print(f"  Found in {len(results)} location(s):")

            for time_offset, song_id in results[:5]:  # Show first 5
                song_info = db_loaded.get_song_info(song_id)
                print(
                    f"    - Song #{song_id} '{song_info['title']}' at t={time_offset:.2f}s"
                )

        # === EXAMPLE 4: Batch query ===
        print("\n" + "=" * 60)
        print("EXAMPLE 4: Batch Query Demo")
        print("=" * 60)

        # Simulate a query with multiple hashes
        # In Phase 5, these will come from a real audio sample
        if db_loaded.hash_table:
            query_hashes = []
            for i, hash_val in enumerate(list(db_loaded.hash_table.keys())[:10]):
                query_hashes.append((hash_val, i * 0.5))  # Mock sample times

            print(f"\nQuerying {len(query_hashes)} hashes...")
            matches = db_loaded.query_hashes(query_hashes)

            print(f"Found matches in {len(matches)} song(s):")
            for song_id, time_pairs in matches.items():
                song_info = db_loaded.get_song_info(song_id)
                print(
                    f"  Song #{song_id} '{song_info['title']}': {len(time_pairs)} matching hashes"
                )

    else:
        print(f"\n⚠ No database found at {DatabaseConfig.DB_FILE}")
        print("Build one first using Example 1")

    print("\n" + "=" * 60)
    print("✅ Phase 4 Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Build your own database with your music collection")
    print("  • Use CLI: python phase4_database.py --dir ./your_folder")
    print("  • Move to Phase 5: Matching Algorithm")
    print("=" * 60 + "\n")
