import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import pandas as pd

from imgaug import augmenters as iaa


EPOCH = 50        #Training epoches
EPOCH_CUSTOM = 10

BATCH_SIZE = 32    #Training batch size
NUM_WORKER = 8     #Training workers
img_size = 224
img_d = ''
img_n = []
 
#############################################################
TRAIN_LR = 0.001   #Learning rate
MOMENTUM = 0.9     #SGD Momemutm
WEIGHT_DECAY = 0.0 #SGD Decay

STEP_SIZE = 20     #Learning rate scheduler
GAMMA = 0.2        #Learning rate scheduler gamma
AGE_STDDEV = 1.0
# DELETE ABOVE WHEN PUBLISHED
#############################################################



def get_dataloaders(base_dir):
    global img_d
    img_d = base_dir + "appa-real-release/"
    # Dataset
    class FaceNPDataset(Dataset):
        def __init__(self, sub='train'):
            if sub != "train" and sub != "test" and sub != "val":
                raise NotImplementedError

            # Loading and sorting labels
            if sub != 'test':
                self.age = np.loadtxt(base_dir + sub + ".txt", delimiter=',')
            else:
                self.age = np.random.rand(500) * 100
            
            self.features = np.load(base_dir + 'feature_' + sub + '.npy')
            assert len(self.age) == len(self.features)

        def __len__(self):
            return len(self.age)

        def __getitem__(self, idx):
            return self.age[idx], self.features[idx]

    train_dataset = FaceNPDataset(sub='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, drop_last=False)

    val_dataset = FaceNPDataset(sub='val')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, drop_last=False)
    
    test_dataset = FaceNPDataset(sub='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, drop_last=False)

    return train_loader, val_loader, test_loader


def get_img_dataloaders(base_dir):
    global img_d
    img_d = base_dir + "appa-real-release/"
    data_dir = img_d
    train_dataset = FaceDataset(data_dir, "train", img_size=224, augment=True,
                          age_stddev=AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKER, drop_last=True)
    val_dataset = FaceDataset(data_dir, "valid", img_size=224, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKER, drop_last=False)
    test_dataset = FaceDataset(data_dir, "test", img_size=224, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKER, drop_last=False)
    return train_loader, val_loader, test_loader


def show_data(base_dir):
    global img_d
    global img_n
    img_d = base_dir
    plt.figure(figsize=(15, 8))
    img_list = os.listdir(img_d + 'valid/')
    img_list = sorted(img_list, key=lambda x: x[0:6])

    cnt = 0
    for name in img_list[::-1]:
        if len(name) == 19:
            img = Image.open(img_d + 'valid/' + name)
            img_n.append(name)
            plt.subplot(1, 6, cnt + 1)
            plt.imshow(img)
            cnt += 1
            if cnt == 6:
                break
                
def show_results(preds, gt):
    plt.figure(figsize=(15,8))
    img_size = 225
    for i in range(6):
        img = cv2.imread(img_d + 'valid/' + img_n[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, str(int(preds[::-1][i])), (0, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(int(gt[::-1][i])), (180, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        plt.subplot(1,6,i+1)
        plt.imshow(img)


def test(model, loader, filename):
    model.eval()
    preds = []
    for i, (y, x) in enumerate(loader):
        x, y = x.cuda().float(), y.cuda().float().reshape(-1,1)
        outputs = model(x)

        preds.append(outputs.cpu().detach().numpy())

    preds = np.concatenate(preds, axis=0)
    np.savetxt(filename, preds, delimiter=',')
    return preds

def test_cel(model, loader, filename):
    model.eval()
    preds = []
    for i, (y,x) in enumerate(loader):
        x= x.cuda().float()
        outputs = model(x)
        preds.append(F.softmax(outputs, dim=-1).cpu().detach().numpy())

    preds = np.concatenate(preds, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    np.savetxt(filename, ave_preds, delimiter=',')
    return ave_preds


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
                ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_dir, data_type, img_size=224, augment=False, age_stddev=1.0):
        assert(data_type in ("train", "valid", "test"))
        csv_path = Path(data_dir).joinpath(f"gt_avg_{data_type}.csv")
        img_dir = Path(data_dir).joinpath(data_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev
        self.data_type = data_type

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        ignore_path = "ignore_list.csv"
        ignore_img_names = list(pd.read_csv(str(ignore_path))["img_name"].values)

        for _, row in df.iterrows():
            img_name = row["file_name"]

            if img_name in ignore_img_names:
                continue

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert(img_path.is_file())
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img).astype(np.float32)
        
        if self.data_type != "test":
            return np.clip(round(age), 0, 100),torch.from_numpy(np.transpose(img, (2, 0, 1)))
        else:
            return np.random.rand(1)*100,torch.from_numpy(np.transpose(img, (2, 0, 1)))

    