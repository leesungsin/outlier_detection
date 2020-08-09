
import torch
import torch.nn as nn
import os
import time
import datetime
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data import FolderImagePerClassDataset, to_img

class Trainer():
    def __init__(self, model, criterion, optimizer, transform=None, target_transform=None, **kwargs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.options = kwargs

    def load_dataloader(self, loader_type, data_type):
        data_loader = None
        # TODO: DataLoader parameters 도 options 에서 할 수 있도록 변경
        if loader_type is "FolderPerClass":
            data_loader = DataLoader(FolderImagePerClassDataset(
                root="",
                transform="",
                kwargs=self.options
            ), batch_size=1, shuffle=True)

        return data_loader

    def train(self):

        # TODO: dataloader 에 들어가는 options 들 configs.py 파일에서 관리할 수 있도록 변경
        train_loader = self.load_dataloader("FolderPerClass","train")
        epochs = "" # self.options["configs"].epochs 를 이리로 옮겨야 하나?
        for epoch in range(self.options["configs"].epochs):
            start = time.time()
            self.model.train()
            loss = 0.0

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.options["device"])
                self.optimizer.zero_grad()
                decode_z, z = self.model(data)
                train_loss = self.criterion(decode_z, data)
                train_loss.backward()
                self.optimizer.step()
                loss += train_loss.item()

                # TODO: loss print 변경, f-string 으로 표현할때 float 자리수를 어떻게?
                if batch_idx % 10 ==0:
                    print(f"epoch [{epoch}/{self.options['configs'].epochs}] loss: {loss}")
                    # TODO: add save_image path in configs.py
                    # visualization for decoded images
                    images = to_img(decode_z.cpu().data)
                    save_image(images, "")

                    # TODO: tensorboard 에 이미지 복원된 결과 visualization?
                    # https://www.tensorflow.org/tensorboard/image_summaries


            loss = loss / len(train_loader)
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
            sec = time.time() - start
            times = str(datetime.timedelta(seconds=sec)).split(".")
            times = times[0]

            print(f"Epoch: {epoch} is end ==============[{times}]================")
