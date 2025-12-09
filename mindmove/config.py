class config:
    def __init__(self):
        # Signal parameters
        self.FSAMP = 2000
        self.fNy = self.FSAMP / 2
        self.DT = 1 / self.FSAMP

        # Channels settings
        self.num_channels = 32
        # self.dead_channels = [1, 6, 16, 26, 27] # 1-indexed
        self.dead_channels = [22] # 1-indexed
        self.active_channels = [i for i in range(self.num_channels) if i not in self.dead_channels]


        # Processing options
        self.ENABLE_FILTERING = True
        # self.ENABLE_FILTERING = False


        # data to open
        self.OPEN_TEMPLATES_FOLDER = "data/19.11.2025/templates/open"
        self.OPEN_GT_FOLDER = "data/19.11.2025/ground truths/open"
        self.CLOSED_TEMPLATES_FOLDER = "data/19.11.2025/templates/closed"
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
        

                

        self.LABELS = {0: 'Open', 1: "Closed"}

        # Decision logic
        self.POST_PREDICTION_SMOOTHING = "MAJORITY VOTE"  # Options: "NONE", "MAJORITY VOTE", "5 CONSECUTIVE"
        self.SMOOTHING_WINDOW = 5


config = config()