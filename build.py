

import torch
import torch.optim as optim
from models import AE
from configs import Config

def build_model(configs):

    model, criterion, optimizer = None, None, None

    if configs.model_name == "AE":
        model = AE(configs.num_in_channels, configs.z_size, configs.num_filters)

    if configs.criterion == "mse":
        criterion = torch.nn.MSELoss()

    if configs.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=configs.optimizer_learning_rate, momentum=configs.optimizer_momentum)

    elif configs.optimizer == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=configs.optimizer_learning_rate, weight_decay=1e-5)

    return model, criterion, optimizer
