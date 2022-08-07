import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import *
from utils import *


def Predict(image, names_path="labels.txt", weights="checkpoints/classifier_latest.ckpt"):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # names
    with open(names_path, "r") as f:
        names = [i.rstrip() for i in f.readlines()]

    # load model
    model = CLASSIFIER(len(names)).to(device)
    model.load_state_dict(torch.load(weights))
    model.eval()
    
    # load image
    img = cv2.imread(image)
    input_img = resize(transforms.ToTensor()(img), 96)
    input_img = Variable(input_img.unsqueeze(0).to(device))

    # predict
    with torch.no_grad():
        detections = model(input_img)[0]
    values, indices = torch.sort(detections, descending=True)

    # top-5 predictions
    display = 5 if len(indices) > 5 else len(indices)
    for i in range(display):
        print( "\t+ %s: %.4f%%" %(names[indices[i]], values[i] * 100))
    return names[indices[0]], float(values[0])

# if __name__ == "__main__":
#     for line in open('./test.txt', 'r').readlines():
#         line = line.strip('\n')
#         print(line + ":    ")
#         predict(line)
