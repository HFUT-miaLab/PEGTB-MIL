import os
import random

import numpy as np
import torch
import torch.utils.data as data

from utils.util import coords_norm


class TrainValDataset(data.Dataset):
    def __init__(self, path, data_csv, mode, fold_k, sample_num=None):
        super(TrainValDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.fold_k = fold_k
        self.sample_num = sample_num
        self.data_list = []
        self.cls_list = []
        with open(data_csv, 'r') as f:
            for i in f.readlines():
                if self.mode == 'train':
                    slide_feature_path = i.split(',')[0]
                    slide_label = int(i.split(',')[1].strip())
                    fold = int(i.split(',')[-1].strip())
                    if fold != self.fold_k:
                        self.data_list.append((slide_feature_path, slide_label))
                        self.cls_list.append(slide_label)
                else:
                    slide_feature_path = i.split(',')[0]
                    slide_label = int(i.split(',')[1].strip())
                    fold = int(i.split(',')[-1].strip())
                    if fold == self.fold_k:
                        self.data_list.append((slide_feature_path, slide_label))
                        self.cls_list.append(slide_label)

    def __getitem__(self, index: int):
        feature_data = torch.load(os.path.join(self.path, self.data_list[index][0] + "_features.pth"))
        coordinates = torch.load(os.path.join(self.path, self.data_list[index][0] + "_coordinates.pth"))
        coordinates = coords_norm(coordinates)
        label = torch.LongTensor([self.data_list[index][1]])

        N, _ = feature_data.shape
        if self.mode == 'train' and self.sample_num is not None:
            if N > self.sample_num:
                select_index = torch.LongTensor(random.sample(range(N), self.sample_num))
            else:
                select_index = torch.LongTensor(random.sample(range(N), N))
            feature_data = torch.index_select(feature_data, 0, select_index)
            coordinates = torch.index_select(coordinates, 0, select_index)
        return feature_data, coordinates, label

    def __len__(self):
        return len(self.data_list)

    def get_weights(self):
        labels = np.array(self.cls_list)
        tmp = np.bincount(labels)
        weights = 1 / np.array(tmp[labels], np.float64)
        return weights

class TestDataset(data.Dataset):
    def __init__(self, path, data_csv):
        super(TestDataset, self).__init__()
        self.path = path
        self.data_list = []
        with open(data_csv, 'r') as f:
            for i in f.readlines():
                slide_feature_path = i.split(',')[0]
                slide_label = int(i.split(',')[-1].strip())
                self.data_list.append((slide_feature_path, slide_label))

    def __getitem__(self, index: int):
        feature_data = torch.load(os.path.join(self.path, self.data_list[index][0] + "_features.pth"))
        coordinates = torch.load(os.path.join(self.path, self.data_list[index][0] + "_coordinates.pth"))
        coordinates = coords_norm(coordinates)
        label = torch.LongTensor([self.data_list[index][1]])

        return feature_data, coordinates, label

    def __len__(self):
        return len(self.data_list)
