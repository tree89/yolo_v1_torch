import torch

from src.model import YOLOV1

# TODO: DEVICE have to be changed with os.env.get
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def train():
    model = YOLOV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

if __name__=="__main__":
    train()
