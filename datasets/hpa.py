from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random
import cv2
from skimage.io import imread
from torchvision.transforms import Compose, ToTensor, Normalize


class HPABaseline(Dataset):
    def __init__(
        self, 
        df, 
        base_tfms=None,
        aug_tfms=None,
        image_path=None,
        image_size=512,
        conf_aware=False,
        conf_path=None,
        mode='train'):

        # Store variables
        self.df = df.reset_index(drop=True)
        self.base_transform = base_tfms
        self.aug_transform = aug_tfms
        self.image_path = image_path
        self.image_size = image_size
        self.conf_aware = conf_aware
        self.conf_path = conf_path
        self.mode = mode

        if conf_aware:
            self.conf_df = pd.read_csv(conf_path)
            self.conf_df = self.conf_df.reset_index(drop=True)
            self.conf_df = self.conf_df.set_index('filename')
            self.conf_cols = ['prob_{}'.format(i) for i in range(19)]
        else:
            self.conf_df = None

        self.cols = ['class{}'.format(i) for i in range(19)]  # Target label column names


    def get_labels(self):
        """ Returns the labels of the dataset as a numpy array. """
        labels = [row[self.cols].values.astype(np.float64) for index, row in self.df.iterrows()]
        return np.array(labels)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        row = self.df.loc[index]

        # -------- TRAIN MODE --------
        if self.mode == 'train':
            # Load image
            path = f'{self.image_path}/{row["ID"]}.png'
            img = imread(path)

            # Apply optional image augmentations
            if self.aug_transform is not None:
                res = self.aug_transform(image=img)
                img = res['image']

            # Ensure image has the correct size
            if not img.shape[0] == self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size))

            # Apply tensor conversion and normalization
            img = self.base_transform(img)

            # Store processed image and metadata
            label = row[self.cols].values.astype(np.float64)
            if self.conf_aware:
                conf_row = self.conf_df.loc[row['ID']]
                conf = conf_row[self.conf_cols].values.astype(np.float64)
                return img, label, conf

            return img, label

        # -------- VALIDATION MODE --------
        if self.mode == 'valid':
            path = f'{self.image_path}/{row["ID"]}.png'
            img = imread(path)

            if self.aug_transform is not None:
                res = self.aug_transform(image=img)
                img = res['image']

            if not img.shape[0] == self.image_size:
                img = cv2.resize(img, (self.image_size, self.image_size))

            img = self.base_transform(img)
            label = row[self.cols].values.astype(np.float64)

            return img, label
            

class ConfAwareHPADataset(Dataset):
    def __init__(
        self, 
        df, 
        base_tfms=None,
        aug_tfms=None,
        cell_path=None,
        cell_count=16,
        cell_size=256,
        conf_aware=False,
        conf_path=None,
        mode='train'):

        # Store variables
        self.df = df.reset_index(drop=True)
        self.base_transform = base_tfms
        self.aug_transform = aug_tfms
        self.cell_path = cell_path
        self.cell_count = cell_count
        self.cell_size = cell_size
        self.conf_aware = conf_aware
        self.conf_path = conf_path
        self.mode = mode

        if conf_aware:
            self.conf_df = pd.read_csv(conf_path)
            self.conf_df = self.conf_df.reset_index(drop=True)
            self.conf_df = self.conf_df.set_index('filename')
            self.conf_cols = ['prob_{}'.format(i) for i in range(19)]
        else:
            self.conf_df = None

        self.cols = ['class{}'.format(i) for i in range(19)]  # Target label column names


    def get_labels(self):
        """ Returns the labels of the dataset as a numpy array. """
        labels = [row[self.cols].values.astype(np.float64) for index, row in self.df.iterrows()]
        return np.array(labels)


    def get_num_cells(self):
        """ Returns the number of cells of each image on the dataset. """
        return self.df['idx']


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        row = self.df.loc[index]

        # -------- TRAIN MODE --------
        if self.mode == 'train':
            if self.cell_count == -1:
                cnt = row['idx']
            else:
                cnt = self.cell_count

            # If more cells than count, sample a subset
            if row['idx'] > cnt:
                selected = random.sample([i for i in range(row['idx'])], cnt)
            else:
                # Otherwise use all available cells
                selected = [i for i in range(row['idx'])]

            # Allocate empty tensors for images, masks, labels and confidence scores
            batch = torch.zeros((cnt, 4, self.cell_size, self.cell_size))
            label = np.zeros((cnt, 19))
            img_label = np.zeros((19))
            conf = np.zeros((cnt, 19))
            img_conf = np.zeros((19))

            # Load and process each selected cell image
            for idx, s in enumerate(selected):
                path = f'{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)

                # Apply optional image augmentations
                if self.aug_transform is not None:
                    res = self.aug_transform(image=img)
                    img = res['image']

                # Ensure image has the correct size
                if not img.shape[0] == self.cell_size:
                    img = cv2.resize(img, (self.cell_size, self.cell_size))

                # Apply tensor conversion and normalization
                img = self.base_transform(img)

                # Store processed image and metadata
                batch[idx, :, :, :] = img
                label[idx] = row[self.cols].values.astype(np.float64)
                if self.conf_aware:
                    conf_row = self.conf_df.loc[row['ID']+f'_{s+1}']
                    conf[idx] = conf_row[self.conf_cols].values.astype(np.float64)

            img_label = row[self.cols].values.astype(np.float64)

            if self.conf_aware:
                img_conf = self.conf_df.loc[row['ID']]
                img_conf = img_conf[self.conf_cols].values.astype(np.float64)

            # Convert values to torch tensors
            # batch = torch.tensor(batch)
            label = torch.tensor(label)
            img_label = torch.tensor(img_label)
            conf = torch.tensor(conf)
            img_conf = torch.tensor(img_conf)
            cnt = torch.tensor(cnt)

            return batch, label, img_label, conf, img_conf, cnt

        # -------- VALIDATION MODE --------
        if self.mode == 'valid':
            selected = [i for i in range(row['idx'])]  # use all cells for validation
            cnt = row['idx']  # number of cells

            batch = torch.zeros((cnt, 4, self.cell_size, self.cell_size))
            label = np.zeros((cnt, 19))

            for idx, s in enumerate(selected):
                path = f'{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)

                if self.aug_transform is not None:
                    res = self.aug_transform(image=img)
                    img = res['image']

                if not img.shape[0] == self.cell_size:
                    img = cv2.resize(img, (self.cell_size, self.cell_size))

                img = self.base_transform(img)

                batch[idx, :, :, :] = img
                label[idx] = row[self.cols].values.astype(np.float64)

            return batch, label, row[self.cols].values.astype(np.float64), cnt

    
class GetPredictionsDataset(Dataset):
    def __init__(self, df, tfms=None, cell_path=None, cell_size=256):
        print('[ i ] GetPredictionsDataset')

        self.df = df.reset_index(drop=True)
        self.transform = tfms
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])
        self.cell_path = cell_path
        self.cell_size = cell_size
        self.cols = ['class{}'.format(i) for i in range(19)]

        print('self.cell_path: {}'.format(self.cell_path))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        selected = [i for i in range(row['idx'])]
        cnt = row['idx']
        filename = row['ID']

        batch = torch.zeros((cnt, 4, self.cell_size, self.cell_size))
        mask = np.zeros((cnt))
        label = np.zeros((cnt, 19))
        for idx, s in enumerate(selected):
            path = f'{self.cell_path}/{row["ID"]}_{s+1}.png'
            img = imread(path)
            if self.transform is not None:
                res = self.transform(image=img)
                img = res['image']
            if not img.shape[0] == self.cell_size:
                img = cv2.resize(img, (self.cell_size, self.cell_size))
            img = self.tensor_tfms(img)
            batch[idx, :, :, :] = img
            mask[idx] = 1
            label[idx] = row[self.cols].values.astype(np.float64)

        return batch, mask, label, row[self.cols].values.astype(np.float64), cnt, filename
    

class NegativeClassifierDataset(Dataset):
    def __init__(
        self, 
        df, 
        tfms=None,
        cell_path=None,
        cell_size=256,
        conf_aware=False,
        conf_path=None,
        mode='train'):

        # Store variables
        self.df = df.reset_index(drop=True)
        self.transform = tfms
        self.cell_path = cell_path
        self.cell_size = cell_size
        self.conf_aware = conf_aware
        self.conf_path = conf_path
        self.mode = mode

        if conf_aware:
            self.conf_df = pd.read_csv(conf_path)
            self.conf_df = self.conf_df.reset_index(drop=True)
            self.conf_df = self.conf_df.set_index('filename')
            self.conf_cols = ['prob_{}'.format(i) for i in range(19)]
        else:
            self.conf_df = None
        
        # Normalization and conversion to tensor
        self.tensor_tfms = Compose([
            ToTensor(),  # Converts image to PyTorch tensor (C x H x W)
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),  # Normalizes each channel
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]

        # -------- TRAIN MODE --------
        if self.mode == 'train':
            if self.conf_aware:
                raise NotImplementedError("NegativeClassifier does not support conf_aware mode.")

            path = f'{self.cell_path}/{row["filename"]}.png'
            img = imread(path)

            # Apply optional image augmentations
            if self.transform is not None:
                res = self.transform(image=img)
                img = res['image']

            # Ensure image has the correct size
            if not img.shape[0] == self.cell_size:
                img = cv2.resize(img, (self.cell_size, self.cell_size))

            # Apply tensor conversion and normalization
            img = self.tensor_tfms(img)
            img_label = torch.tensor(row['is_negative'])

            return img, img_label

        # -------- VALIDATION MODE --------
        if self.mode == 'valid':
            path = f'{self.cell_path}/{row["filename"]}.png'
            img = imread(path)

            # Apply optional image augmentations
            if self.transform is not None:
                res = self.transform(image=img)
                img = res['image']

            # Ensure image has the correct size
            if not img.shape[0] == self.cell_size:
                img = cv2.resize(img, (self.cell_size, self.cell_size))

            # Apply tensor conversion and normalization
            img = self.tensor_tfms(img)
            img_label = torch.tensor(row['is_negative'])

            return img, img_label
        

class SimCLRDataset(Dataset):
    def __init__(
        self, 
        df, 
        base_tfms=None,
        aug_tfms=None,
        cell_path=None,
        cell_size=256):

        # Store variables
        self.df = df.reset_index(drop=True)
        self.base_transform = base_tfms
        self.aug_transform = aug_tfms
        self.cell_path = cell_path
        self.cell_size = cell_size

        self.cols = ['class{}'.format(i) for i in range(19)]  # Target label column names

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        path = f'{self.cell_path}/{row["ID"]}.png'
        img = imread(path)

        # Apply optional image augmentations
        res1 = self.aug_transform(image=img)
        img1 = res1['image']
        res2 = self.aug_transform(image=img)
        img2 = res2['image']

        # Ensure image has the correct size
        if not img1.shape[0] == self.cell_size:
            img1 = cv2.resize(img1, (self.cell_size, self.cell_size))
        if not img2.shape[0] == self.cell_size:
            img2 = cv2.resize(img2, (self.cell_size, self.cell_size))
        
        # Apply tensor conversion and normalization
        img1 = self.base_transform(img1)
        img2 = self.base_transform(img2)

        # Label
        labels = torch.tensor(row[self.cols].values.astype(np.float64))

        return ((img1, img2), labels)


  