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
    TARGET_T_MIN = 0.0
    TARGET_T_MAX = 2.0
    TARGET_F_MIN = 0
    TARGET_F_MAX = 1000

    # Fan-out: max number of target points per anchor
    FAN_OUT = 10

    # Hash packing
    FREQ_BITS = 10
    TIME_BITS = 10


class DatabaseConfig:
    """Configuration for database storage"""

    # Storage paths
    DB_FILE = "./utils/fingerprint_database.pkl"
    METADATA_FILE = "./utils/song_metadata.json"


class MatchConfig:
    """Configuration for matching algorithm"""

    # Matching thresholds
    MIN_MATCHES = 5

    # Histogram binning
    BIN_SIZE = 0.5

    # Confidence scoring
    CONFIDENCE_THRESHOLD = 5
