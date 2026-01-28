class config:
    def __init__(self):
        # Signal parameters
        self.FSAMP = 2000
        self.fNy = self.FSAMP / 2
        self.DT = 1 / self.FSAMP

        # Channels settings
        self.num_channels = 32
        # dead_channels uses 0-indexed values internally (0-31 for 32 channels)
        # In the UI, users enter 1-indexed values (1-32), which are converted to 0-indexed
        # Example: channel 22 (1-indexed user input) = index 21 (0-indexed internal)
        # self.dead_channels = [0, 5, 15, 25, 26]  # channels 1, 6, 16, 26, 27 (1-indexed)
        # self.dead_channels = [21]  # channel 22 (1-indexed) = index 21 (0-indexed)
        self.dead_channels = []
        # self.dead_channels = [8, 21, 24]  # channels 9, 22, 25 (1-indexed)

        self.active_channels = [i for i in range(self.num_channels) if i not in self.dead_channels]


        # Processing options
        self.ENABLE_FILTERING = True  # Default state of filter toggle in device interface
        self.ENABLE_DIFFERENTIAL_MODE = False  # Default: monopolar (32 ch), True: single differential (16 ch)

        # === FILTERING CONFIGURATION ===
        # Bandpass filter settings
        self.FILTER_LOWCUT = 20      # High-pass cutoff (Hz) - removes motion artifacts
        self.FILTER_HIGHCUT = 500    # Low-pass cutoff (Hz) - EMG bandwidth
        self.FILTER_ORDER = 4        # Butterworth filter order

        # Notch filter settings (powerline noise)
        self.NOTCH_FREQ = 50         # Fundamental frequency (50 Hz EU, 60 Hz US)
        self.NOTCH_HARMONICS = 8     # Number of harmonics to filter (50, 100, ... 400 Hz)
        self.NOTCH_Q = 30            # Quality factor (higher = narrower notch)

        # Diagnostic mode
        self.DIAGNOSTIC_MODE = False

        # DTW implementation selection
        # Options: USE_NUMBA_DTW (fastest CPU), USE_TSLEARN_DTW (Euclidean), USE_GPUDTW (GPU-accelerated)
        self.USE_NUMBA_DTW = True   # Use numba JIT-compiled DTW (fastest CPU, cosine distance)
        self.USE_TSLEARN_DTW = False  # Use tslearn DTW (Euclidean distance)
        self.USE_GPUDTW = False  # Use GPU-accelerated DTW (requires CUDA/OpenCL, Euclidean distance)


        # data to open
        self.OPEN_TEMPLATES_FOLDER = "data/templates_open"
        self.OPEN_GT_FOLDER = "data/19.11.2025/ground truths/open"
        self.CLOSED_TEMPLATES_FOLDER = "data/templates_closed"
        self.CLOSED_GT_FOLDER = "data/19.11.2025/ground truths/closed"
        self.TEST_FOLDER = "data/19.11.2025/templates/test"
        self.TEST_GT_FOLDER = "data/19.11.2025/ground truths/test"

        # DTW setup
        # dead channels removal

        # ######## PAY ATTENTION THE WINDOWS AND OVERLAP CHOSEN FOR THE **TRAINING** RECORDINGS

        self.window_length =int(0.096 * self.FSAMP) # s x FSAMP #length of the window in which the 1 second templates are divided to extract the features
        self.increment = int(0.032 * self.FSAMP) # s x FSAMP
        self.template_duration = 1 # s
        self.template_nsamp = self.template_duration*self.FSAMP # number of samples in the template length chosen
        self.increment_dtw = 0.05 # s
        self.increment_dtw_samples = int(self.FSAMP * self.increment_dtw)

        # Template extraction settings
        self.MIN_ACTIVATION_DURATION_S = 1.5  # Minimum activation duration to extract (seconds)
        self.TARGET_TEMPLATES_PER_CLASS = 20  # Target number of templates per class
        self.KINEMATICS_FSAMP = 60  # Virtual hand kinematics sampling frequency (Hz)
        self.ONSET_OFFSET_S = 0.2  # Seconds before GT=1 to start cutting (for onset capture)
        self.HOLD_SKIP_S = 0.5  # Seconds to skip after GT=1 for hold-only templates

        self.LABELS = {0: 'Open', 1: "Closed"}

        # Decision logic
        self.POST_PREDICTION_SMOOTHING = "MAJORITY VOTE"  # Options: "NONE", "MAJORITY VOTE", "5 CONSECUTIVE"
        self.SMOOTHING_WINDOW = 5


config = config()