

from configs import Config
import torch
from build import build_model
from trainer import Trainer

def main():
    configs = Config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, criterion, optimizer = build_model(configs)
    model = model.to(device)
    trainer = Trainer(model, criterion, optimizer, configs=configs, device=device)
    trainer.train()



if __name__ == '__main__':
    main()