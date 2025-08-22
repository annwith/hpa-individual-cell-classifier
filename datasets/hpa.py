from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import random
import cv2
from skimage.io import imread
from torchvision.transforms import Compose, ToTensor, Normalize


class HPADataset(Dataset):
    NAME: str = "hpa"

    def __init__(
        self, 
        df, 
        tfms=None,
        cell_path=None,
        cell_count=16,
        cell_size=256,
        label_smoothing=0,
        mode='train'):

        """
        Custom PyTorch dataset for loading images and labels from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing image filenames and labels.
            tfms (callable, optional): Transformations to apply to images.
            cell_path (str): Path to the directory containing cell images.
            cell_count (int): Number of cells to sample from each image.
            cell_size (int): Size of the cell images.
            mode (str): Mode of operation ('train' or 'valid').
        """

        # Store DataFrame and reset index for consistency
        self.df = df.reset_index(drop=True)
        self.transform = tfms
        self.cell_path = cell_path
        self.cell_count = cell_count
        self.cell_size = cell_size
        self.label_smoothing = label_smoothing
        self.mode = mode

        # Define image normalization (expects 4 channels)
        self.tensor_tfms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406, 0.406], std=[0.229, 0.224, 0.225, 0.225]),
        ])

        # Define label columns (19 classes)
        self.cols = ['class{}'.format(i) for i in range(19)]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, index):
        """
        Loads an image, applies transformations, and returns image tensor with labels.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (batch, mask, label, raw_label) for training
                   (batch, mask, label, raw_label, cnt) for validation
        """
        # Handle training samples
        if self.mode == 'train':
            row = self.df.loc[index]  # Get row from DataFrame
            cnt = self.cell_count  # Max number of images per sample

            # Select a subset of images if more are available
            if row['idx'] > cnt:
                selected = random.sample([i for i in range(row['idx'])], cnt)
            else:
                selected = [i for i in range(row['idx'])]

            # Initialize batch tensors
            batch = torch.zeros((cnt, 4, self.cell_size, self.cell_size))
            mask = np.zeros((cnt))  # Binary mask (1 if image is present)
            label = np.zeros((cnt, 19))  # Label array for the batch

            # Iterate over selected images
            for idx, s in enumerate(selected):
                path = f'{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)  # Read image

                # Apply transformations if provided
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']

                # Resize if necessary
                if not img.shape[0] == self.cell_size:
                    img = cv2.resize(img, (self.cell_size, self.cell_size))

                # Convert image to tensor and normalize
                img = self.tensor_tfms(img)
                batch[idx, :, :, :] = img  # Store in batch tensor
                mask[idx] = 1  # Mark image as present
                label[idx] = row[self.cols].values.astype(np.float64)  # Store label

            # Apply label smoothing if specified
            if self.label_smoothing == 0:
                return batch, mask, label, row[self.cols].values.astype(np.float64)
            else:
                return batch, mask, 0.9 * label + 0.1 / 19, 0.9 * row[self.cols].values.astype(np.float64) + 0.1 / 19

        # Handle validation samples
        if self.mode == 'valid':
            row = self.df.loc[index]
            selected = [i for i in range(row['idx'])]  # Use all available images
            cnt = row['idx']  # Number of images

            # Initialize batch tensors
            batch = torch.zeros((cnt, 4, self.cell_size, self.cell_size))
            mask = np.zeros((cnt))
            label = np.zeros((cnt, 19))

            # Iterate over images
            for idx, s in enumerate(selected):
                path = f'{self.cell_path}/{row["ID"]}_{s+1}.png'
                img = imread(path)

                # Apply transformations if provided
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']

                # Resize if necessary
                if not img.shape[0] == self.cell_size:
                    img = cv2.resize(img, (self.cell_size, self.cell_size))

                # Convert image to tensor and normalize
                img = self.tensor_tfms(img)
                batch[idx, :, :, :] = img
                mask[idx] = 1
                label[idx] = row[self.cols].values.astype(np.float64)

            return batch, mask, label, row[self.cols].values.astype(np.float64), cnt


class ConfAwareHPADataset(Dataset):
    NAME: str = "hpa"

    def __init__(
        self, 
        df, 
        tfms=None,
        cell_path=None,
        cell_count=16,
        cell_size=256,
        conf_aware=False,
        conf_path=None,
        label_smoothing=0,
        mode='train'):

        # Store variables
        self.df = df.reset_index(drop=True)
        self.transform = tfms
        self.cell_path = cell_path
        self.cell_count = cell_count
        self.cell_size = cell_size
        self.conf_aware = conf_aware
        self.conf_path = conf_path
        self.label_smoothing = label_smoothing
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

        self.cols = ['class{}'.format(i) for i in range(19)]  # Target label column names

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
                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']

                # Ensure image has the correct size
                if not img.shape[0] == self.cell_size:
                    img = cv2.resize(img, (self.cell_size, self.cell_size))

                # Apply tensor conversion and normalization
                img = self.tensor_tfms(img)

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

            # Apply label smoothing if configured
            if self.label_smoothing == 0:
                return batch, label, img_label, conf, img_conf, cnt
            else:
                label = 0.9 * label + 0.1 / 19
                img_label = 0.9 * img_label + 0.1 / 19
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

                if self.transform is not None:
                    res = self.transform(image=img)
                    img = res['image']

                if not img.shape[0] == self.cell_size:
                    img = cv2.resize(img, (self.cell_size, self.cell_size))

                img = self.tensor_tfms(img)

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
        label_smoothing=0,
        mode='train'):

        # Store variables
        self.df = df.reset_index(drop=True)
        self.transform = tfms
        self.cell_path = cell_path
        self.cell_size = cell_size
        self.conf_aware = conf_aware
        self.conf_path = conf_path
        self.label_smoothing = label_smoothing
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

            # Apply label smoothing if configured
            if self.label_smoothing == 0:
                return img, img_label
            else:
                raise NotImplementedError("Label smoothing is not implemented for NegativeClassifier in train mode.")

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
        