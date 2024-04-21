import glob
import cv2
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, mode="train", device="cuda"):
        self.path = "./dataset/"
        self.device = device
        self.filenames_list = glob.glob(self.path+"*.png")
        self.transform = transform

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.filenames_list[idx]
        image = cv2.imread(img_path)
        image = image/255.0
        if self.transform:
            image = self.transform(image=image)["image"]            
        return image.to(self.device), img_path