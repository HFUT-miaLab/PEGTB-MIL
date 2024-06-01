import sys
import datetime
import random

import torch
import numpy as np


class Logger(object):
    def __init__(self, filename='./logs/' + datetime.datetime.now().strftime('%Y-%m-%d %H%M%S.%f') + '.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class BestModelSaver:
    def __init__(self):
        self.best_valid_acc = 0
        self.best_valid_auc = 0
        self.best_valid_f1 = 0
        self.best_valid_acc_epoch = 0
        self.best_valid_auc_epoch = 0
        self.best_valid_f1_epoch = 0


    def update(self, valid_acc, valid_auc, valid_f1, current_epoch):

        if valid_acc >= self.best_valid_acc:
            self.best_valid_acc = valid_acc
            self.best_valid_acc_epoch = current_epoch
        if valid_auc >= self.best_valid_auc:
            self.best_valid_auc = valid_auc
            self.best_valid_auc_epoch = current_epoch
        if valid_f1 >= self.best_valid_f1:
            self.best_valid_f1 = valid_f1
            self.best_valid_f1_epoch = current_epoch


def fix_random_seeds(seed=None):
    """
    Fix random seeds.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Fix Random Seeds:", seed)


def merge_config_to_args(args, cfg):
    # Data
    args.feature_root = cfg.DATA.FEATURE_ROOT
    args.train_valid_csv = cfg.DATA.TRAIN_VALID_CSV
    args.test_csv = cfg.DATA.TEST_CSV

    # Model
    args.feat_dim = cfg.MODEL.FEATURE_DIM
    args.num_class = cfg.MODEL.NUM_CLASS
    args.mask_ratio = cfg.MODEL.MASK_RATIO
    args.pos_dim = cfg.MODEL.POS_DIM
    args.later_dim = cfg.MODEL.LATER_DIM
    args.loss_weight = cfg.MODEL.LOSS_WEIGHT
    args.return_atte = cfg.MODEL.RETURN_ATTE
    args.return_pred_coordinates = cfg.MODEL.RETURN_PRED_COORDINATES
    args.return_pos_error = cfg.MODEL.RETURN_POS_ERROR

    # TRAIN
    args.batch_size = cfg.TRAIN.BATCH_SIZE
    args.workers = cfg.TRAIN.WORKERS
    args.lr = cfg.TRAIN.LR
    args.weight_decay = cfg.TRAIN.WEIGHT_DECAY
    args.max_epoch = cfg.TRAIN.MAX_EPOCH
    args.weights_save_path = cfg.TRAIN.WEIGHTS_SAVE_PATH

def coords_norm(coords):
    max_coords = torch.max(coords, 0, keepdim=True).values
    min_coords = torch.min(coords, 0, keepdim=True).values
    coords = coords - min_coords
    div_coords = max_coords - min_coords
    coords = torch.div(coords, div_coords)
    if div_coords[:, 0] == div_coords[:, 1]:
        return coords
    elif div_coords[:, 0] > div_coords[:, 1]:
        scale = div_coords[:, 1] / div_coords[:, 0]
        coords[:, 1] = coords[:, 1] * scale
        return coords
    elif div_coords[:, 1] > div_coords[:, 0]:
        scale = div_coords[:, 0] / div_coords[:, 1]
        coords[:, 0] = coords[:, 0] * scale
        return coords
    else:
        raise Exception("pos error")
