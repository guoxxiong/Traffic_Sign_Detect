import os
import random

import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import *
from utils import *


def train(names_path="labels.txt", train_path="train.txt", weights=None):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # parameters
    batch = 256
    subdivisions = 1
    learning_rate = 1e-2
    current_batches, max_batches = 0, 100000

    # names
    with open(names_path, "r") as f:
        names = [i.rstrip() for i in f.readlines() if i is not ""]

    # train labels
    with open(train_path, "r") as f:
        train_images = [i.rstrip() for i in f.readlines()]

    # load model
    model = CLASSIFIER(len(names)).to(device)
    model.apply(weights_init_normal)

    if weights is not None and current_batches > 0:
        model.load_state_dict(torch.load(weights %(current_batches)))
        # model.load_state_dict(torch.load("checkpoints/classifier_latest.ckpt"))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    while current_batches < max_batches:
        input_size = random.choice([80, 96, 112])
        loss_value = 0
        optimizer.step()
        optimizer.zero_grad()

        for _ in range(subdivisions):
            imgs, targets = [], []
            for _ in range(batch // subdivisions):
                img_path = random.choice(train_images)

                img = cv2.imread(img_path)
                img = random_reinforcement(img)

                img = transforms.ToTensor()(img)
                img = resize(img, input_size)
                imgs.append(img.unsqueeze(0))

                index = 0
                while index < len(names) and names[index] not in img_path:
                    index += 1
                if index == len(names):
                    print("training error: %s" %(img_path))
                    continue
                target = torch.zeros((1), dtype=torch.long)
                target[0] = index
                targets.append(target)

            imgs = torch.cat(imgs, dim=0)
            targets = torch.cat(targets, dim=0)

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device))

            outputs, loss = model(imgs, targets)
            loss /= subdivisions
            loss.backward()
            loss_value += loss.item()

        current_batches += 1
        print("\t+ batch: %d, loss: %.5f" %(current_batches, loss_value))
        if current_batches % 100 == 0:
            torch.save(model.state_dict(), "checkpoints/classifier_%d.ckpt" %(current_batches))
            torch.save(model.state_dict(), "checkpoints/classifier_latest.ckpt")

if __name__ == "__main__":
    train()
