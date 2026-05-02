from model import IDCClassifier as TheModel
from train import train_model as the_trainer
from predict import predict_image as the_predictor
from dataset import IDCDataset as TheDataset
from dataset import the_dataloader as the_dataloader
from config import batchsize as the_batch_size
from config import epochs as total_epochs

if __name__ == "__main__":
    print("interface.py ready ✓")
    print(f"  TheModel      : {TheModel}")
    print(f"  the_trainer   : {the_trainer.__name__}")
    print(f"  the_predictor : {the_predictor.__name__}")
    print(f"  TheDataset    : {TheDataset}")
    print(f"  the_batch_size: {the_batch_size}")
    print(f"  total_epochs  : {total_epochs}")
