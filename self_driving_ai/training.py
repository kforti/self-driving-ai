import copy
import os
import time
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from skimage.io import imread

from self_driving_ai.utils import *


"""
Credit: https://www.kaggle.com/gxkok21/resnet50-with-pytorch
"""

class DrivingDataset(torch.utils.data.Dataset):
    """
    This is our custom dataset class which will load the images, perform transforms on them,
    and load their corresponding labels.
    """

    def __init__(self, img_dir, labels_csv_file=None, transform=None):
        self.img_dir = img_dir

        if labels_csv_file:
            self.labels_df = pd.read_csv(labels_csv_file)
        else:
            self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]

        self.transform = transform

    def __getitem__(self, idx):
        try:
            img_path = self.labels_df.iloc[idx, 0]
        except AttributeError:
            img_path = self.images[idx]

        #         print("img_path:", img_path)
        img = imread(img_path)

        if self.transform:
            img = self.transform(img)

        sample = {
            "image": img,
        }
        try:
            sample["label"] = self.labels_df.iloc[idx, 1]#torch.tensor((self.labels_df.iloc[idx, 1], self.labels_df.iloc[idx, 2]))
            sample["id"] = idx#self.labels_df.loc[idx, "id"]
        except AttributeError:
            #sample["id"] = os.path.basename(self.images[idx]).replace(".tif", "")
            pass

        return sample

    def __len__(self):
        try:
            return self.labels_df.shape[0]
        except AttributeError:
            return len(self.images)


if __name__ == '__main__':
    # Train
    EPOCHS = 20
    USE_GPU = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if USE_GPU else "cpu")
    writer = SummaryWriter("runs/self_driving_ai")

    IMG_DIR = "../Data/Training_Images"
    LABELS_PATH = "../Data/Training_Data"
    labels_df = pd.read_csv(LABELS_PATH, header=None)
    train_indices, test_indices = train_test_split(labels_df.index - 1, test_size=0.20)

    train_dataset = DrivingDataset(IMG_DIR, LABELS_PATH, transform_pipe)

    model = models.resnet50(pretrained=True)

    # Freeze model weights
    # for param in model.parameters():
    #     param.requires_grad = False
    freeze_layes = 6
    for i, child in enumerate(model.children()):
        if i <= freeze_layes:
            for param in child.parameters():
                param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Linear(
            in_features=2048,
            out_features=1
        ),
    )
    model.to(device)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(
            train_indices
        ))
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(
            test_indices
        ))

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_mse_loss = 0.0

    phases = OrderedDict([("train", train_loader), ("test", test_loader)])

    start = time.time()
    for i in range(EPOCHS):
        epoch = i + 1
        samples = 0
        mse_loss_sum = 0
        correct_sum = 0
        for phase, loader in phases.items():
            for j, batch in enumerate(loader):
                X = batch["image"]
                labels = batch["label"]
                if USE_GPU:
                    X = X.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    y = model(X)#.view(1, 3, 224, 224))
                    loss = criterion(
                        y,
                        labels.view(-1, 1).float()#.float()#
                    )

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    mse_loss_sum += loss.item() * X.shape[0]  # We need to multiple by batch size as loss is the mean loss of the samples in the batch
                    samples += X.shape[0]

                    # Print batch statistics every 50 batches
                    if j % 50 == 49 and phase == "train":
                        print("{}:{} - MSE_loss: {}".format(
                            i + 1,
                            j + 1,
                            float(mse_loss_sum) / float(samples)
                        ))

            # Print epoch statistics
            epoch_mse_loss = float(mse_loss_sum) / float(samples)
            print("epoch: {} - {} MSE_loss:{:.4f}".format(i + 1, phase, epoch_mse_loss))

            # Deep copy the model
            if phase == "test" and epoch_mse_loss > best_epoch_mse_loss:
                writer.add_scalar('training MSE loss', mse_loss_sum / len(train_indices), epoch)
                best_epoch_mse_loss = epoch_mse_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "resnet50-optimal.pth")
    writer.close()

    end = time.time()
    train_time = end - start
    print("Total Training Time: {} seconds".format(train_time))
    print("Training Time Per Epoch: {} seconds".format(train_time / EPOCHS))

