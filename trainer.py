
import torch
import torch.nn as nn
import os
import time
import datetime
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from data import FolderImagePerClassDataset, to_img
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchsummary import summary

class Trainer():
    def __init__(self, model, criterion, optimizer, transform=None, target_transform=None, **kwargs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.options = kwargs
        self.transform = self._transforms(transform)
        self.target_transform = target_transform

    def load_dataloader(self, loader_type, data_type):
        data_loader = None
        if loader_type == "FolderPerClass":
            data_loader = DataLoader(FolderImagePerClassDataset(
                root=os.path.join(self.options["configs"].root_dir, data_type),
                transform=self.transform,
                kwargs=self.options
            ), batch_size=self.options["configs"].batch_size, shuffle=self.options["configs"].data_loader_shuffle, drop_last=True)

        return data_loader

    def _transforms(self, transform):
        if all([transform is None, self.options["configs"].dataset == "MNIST"]):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

    def train(self):

        train_loader = self.load_dataloader(self.options["configs"].data_loader_type, "train")
        epochs = self.options["configs"].epochs
        # TODO: writer 위치 변경
        writer = SummaryWriter()
        for epoch in range(epochs):
            start = time.time()
            self.model.train()
            loss = 0.0

            for batch_idx, (data, target, file_names) in enumerate(train_loader):
                data = data.to(self.options["device"])
                self.optimizer.zero_grad()
                decode_z, z = self.model(data)
                train_loss = self.criterion(decode_z, data)
                train_loss.backward()
                self.optimizer.step()
                # loss += train_loss.item()

                if batch_idx % 10 ==0:
                    print(f"epoch [{epoch}/{self.options['configs'].epochs}] batch[{batch_idx}/{len(train_loader.batch_sampler)}] loss: {train_loss.item():.5f}")
                    # visualization for decoded images
                    images = to_img(decode_z.cpu().data)
                    save_path = os.path.join(self.options["configs"].root_dir, "results")
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    # save_path = os.path.join(self.options["configs"].root_dir, "results" , f"epoch:{epoch}-batch_idx{batch_idx}-image.png")
                    save_image(images, os.path.join(save_path, f"epoch:{epoch}-batch_idx{batch_idx}-image.png"))

                    # TODO: tensorboard 에 이미지 복원된 결과 visualization?
                    # https://www.tensorflow.org/tensorboard/image_summaries
                    grid = torchvision.utils.make_grid(images)
                    writer.add_image("decoded images", grid, 0)
                    writer.add_graph(self.model, images.to(self.options["device"]))

            # loss = loss / len(train_loader)
            # print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
            sec = time.time() - start
            times = str(datetime.timedelta(seconds=sec)).split(".")
            times = times[0]


            print(f"Epoch: {epoch} is end ==============[{times}]================")
        writer.close()