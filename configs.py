


class Config:
    def __init__(self):
        self.num_in_channels = 32
        self.z_size = 100
        self.num_filters = 32

        self.model_name = "AE"
        self.criterion = "mse"
        self.optimizer = "adamW"
        self.optimizer_learning_rate = 0.001
