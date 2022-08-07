import os

import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import *
from utils import *


def test(names_path="labels.txt", test_path="test.txt", weights="checkpoints/classifier_latest.ckpt"):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("error", exist_ok=True)

    # parameters
    batch = 256

    # names
    with open(names_path, "r") as f:
        names = [i.rstrip() for i in f.readlines()]

    # test images
    with open(test_path, "r") as f:
        test_images = [i.rstrip() for i in f.readlines()]

    # load model
    model = CLASSIFIER(len(names)).to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()

    # test
    cnt, n = 0, len(test_images)
    while len(test_images) > 0:
        if len(test_images) > batch:
            test_batch = test_images[:batch]
            del test_images[:batch]
        else:
            test_batch = test_images[:]
            del test_images[:]
        
        imgs, targets = [], []
        for img_path in test_batch:
            img = cv2.imread(img_path)
            img = transforms.ToTensor()(img)
            img = resize(img, 96)
            imgs.append(img.unsqueeze(0))

            index = 0
            while index < len(names) and names[index] not in img_path:
                index += 1
            targets.append(index)
        input_imgs = torch.cat(imgs, dim=0)
        input_imgs = Variable(input_imgs.to(device))
        
        with torch.no_grad():
            detections = model(input_imgs)
        _, indices = torch.sort(detections, dim=1, descending=True)

        for i, index in enumerate(targets):
            if index == indices[i][0]:
                cnt += 1
            # else:
            #     file_name = test_batch[i].split("/")[-1]
            #     os.system("cp test/%s error/%s_%s" %(file_name, names[indices[i][0]], file_name))
    print("%d / %d = %.3f%%" %(cnt, n, cnt / n * 100))


if __name__ == "__main__":
    test()
