


class Config:
    def __init__(self):
        self.num_in_channels = 1
        self.z_size = 100
        self.num_filters = 32

        self.model_name = "AE"
        self.criterion = "mse"
        self.optimizer = "adamW"
        self.optimizer_learning_rate = 0.001

        self.root_dir = "/mnt/aistudionas/datasets/MNIST/images/3_run_data/3th"

        self.data_loader_type = "FolderPerClass"
        self.data_loader_shuffle = True
        self.epochs = 10
        self.dataset = "MNIST"
        self.batch_size = 32
