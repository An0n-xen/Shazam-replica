class AudioConfig:
    """Configuration parameters for fingerprinting"""

    # Audio processing
    SAMPLE_RATE = 22050

    # Spectrogram parameters
    FFT_WINDOW_SIZE = 4096
    OVERLAP_RATIO = 0.5

    # Peak detection parameters
    PEAK_NEIGHBORHOOD_SIZE = 20
    MIN_AMPLITUDE_PERCENTILE = 80
    PEAKS_PER_TIME_WINDOW = 5
    TIME_WINDOW_SEC = 0.5


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


class DatabaseConfig:
    """Configuration for database storage"""

    # Storage paths
    DB_FILE = "fingerprint_database.pkl"  # Pickled database
    METADATA_FILE = "song_metadata.json"  # Song information
