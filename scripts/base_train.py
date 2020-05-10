from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import os.path
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm

from pathlib import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

_project_folder_ = os.path.abspath('../')
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from data.quickdraw_dataset import QuickDrawDataset
from data.tuberlin_dataset import TUBerlinDataset
from models.modelzoo import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_r2cnn import SketchR2CNN
from neuralline.rasterize import Raster


DATASETS = {'tuberlin': TUBerlinDataset, 'quickdraw': QuickDrawDataset}


def train_data_collate(batch):
    length_list = [len(item['points3']) for item in batch] 
    max_length = max(length_list) 

    points3_padded_list = list()
    points3_offset_list = list()
    intensities_list = list()
    category_list = list()
    for item in batch:
        points3 = item['points3']
        points3_length = len(points3)
        points3_padded = np.zeros((max_length, 3), np.float32)
        points3_padded[:, 2] = np.ones((max_length,), np.float32)
        points3_padded[0:points3_length, :] = points3
        points3_padded_list.append(points3_padded)

        points3_offset = np.copy(points3_padded)
        points3_offset[1:points3_length, 0:2] = points3[1:, 0:2] - points3[:points3_length - 1, 0:2]
        points3_offset_list.append(points3_offset)

        intensities = np.zeros((max_length,), np.float32)
        intensities[:points3_length] = 1.0 - np.arange(points3_length, dtype=np.float32) / float(points3_length - 1)
        intensities_list.append(intensities)

        category_list.append(item['category'])

    batch_padded = {
        'points3': points3_padded_list,
        'points3_offset': points3_offset_list,
        'points3_length': length_list,
        'intensities': intensities_list,
        'category': category_list
    }

    sort_indices = np.argsort(-np.array(length_list))
    batch_collate = dict()
    for k, v in batch_padded.items():
        sorted_arr = np.array([v[idx] for idx in sort_indices])
        batch_collate[k] = torch.from_numpy(sorted_arr)
    return batch_collate


class BaseTrain(object):

    def __init__(self):
        self.config = self._parse_args()

        self.modes = ['train', 'valid']
        self.step_counters = {m: 0 for m in self.modes}
        self.reporter = None

        self.device = torch.device('cuda:{}'.format(self.config['gpu']) if torch.cuda.is_available() else 'cpu')
        print('[*] Using device: {}'.format(self.device))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.reporter:
            self.reporter.close()

    def _parse_args(self):
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument('--batch_size', type=int, default=48)
        arg_parser.add_argument('--ckpt_nets', nargs='*')
        arg_parser.add_argument('--ckpt_prefix', type=str, default=None)
        arg_parser.add_argument('--dataset_fn', type=str, default=None)
        arg_parser.add_argument('--dataset_root', type=str, default=None)
        arg_parser.add_argument('--gpu', type=int, default=0)
        arg_parser.add_argument('--imgsize', type=int, default=224)
        arg_parser.add_argument('--learn_rate_step', type=int, default=-1)
        arg_parser.add_argument('--learn_rate', type=float, default=0.0001)
        arg_parser.add_argument('--log_dir', type=str, default=None)
        arg_parser.add_argument('--max_points', type=int, default=321)
        arg_parser.add_argument('--model_fn', type=str, default=None)
        arg_parser.add_argument('--note', type=str, default='')
        arg_parser.add_argument('--num_epochs', type=int, default=1)
        arg_parser.add_argument('--report_hist_freq', type=int, default=100)
        arg_parser.add_argument('--report_image_freq', type=int, default=100)
        arg_parser.add_argument('--report_scalar_freq', type=int, default=100)
        arg_parser.add_argument('--save_epoch_freq', type=int, default=1)
        arg_parser.add_argument('--save_step_freq', type=int, default=-1)
        arg_parser.add_argument('--seed', type=int, default=10)
        arg_parser.add_argument('--thickness', type=float, default=1.0)
        arg_parser.add_argument('--valid_freq', type=int, default=1)
        arg_parser.add_argument('--weight_decay', type=float, default=-1)

        arg_parser = self.add_args(arg_parser)
        config = vars(arg_parser.parse_args())

        config['imgsize'] = CNN_IMAGE_SIZES[config['model_fn']]
        if config['dataset_fn'] == 'quickdraw':
            config['max_points'] = 321
            config['report_image_freq'] = 500
            config['save_epoch_freq'] = 1
            config['valid_freq'] = 1
        elif config['dataset_fn'] == 'tuberlin':
            config['max_points'] = 448
            config['report_image_freq'] = 100
            config['save_epoch_freq'] = 20
            config['valid_freq'] = 20
        else:
            raise Exception('Not valid dataset name!')

        if config['log_dir'] is None:
            raise Exception('No log_dir specified!')
        else:
            if not os.path.exists(config['log_dir']):
                os.makedirs(config['log_dir'], 0o777)

        if config['dataset_root'] is None:
            raise Exception('No dataset_root specified!')

        if config['ckpt_prefix'] is not None:
            if len(config['ckpt_prefix']) < 1:
                config['ckpt_prefix'] = None
        if config['ckpt_nets'] is not None:
            if len(config['ckpt_nets']) < 1:
                config['ckpt_nets'] = None

        if config['seed'] is None:
            config['seed'] = random.randint(0, 2**31 - 1)
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

        with open(os.path.join(config['log_dir'], 'options.json'), 'w') as fh:
            fh.write(json.dumps(config, sort_keys=True, indent=4))

        return config

    def add_args(self, arg_parser):
        return arg_parser

    def run_name(self):
        return None

    def checkpoint_prefix(self):
        return self.config['ckpt_prefix']

    def prepare_dataset(self, dataset_dict):
        pass

    def create_data_loaders(self, dataset_dict):
        raise NotImplementedError

    def create_model(self, num_categories):
        raise NotImplementedError

    def weight_decay_excludes(self):
        return ['bias']

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        raise NotImplementedError

    def run(self):
        dataset_fn = self.config['dataset_fn']
        dataset_root = self.config['dataset_root']
        learn_rate = self.config['learn_rate']
        learn_rate_step = self.config['learn_rate_step']
        log_dir = self.config['log_dir']
        model_fn = self.config['model_fn']
        num_epochs = self.config['num_epochs']
        report_scalar_freq = self.config['report_scalar_freq']
        save_epoch_freq = self.config['save_epoch_freq']
        save_step_freq = self.config['save_step_freq']
        valid_freq = self.config['valid_freq']
        weight_decay = self.config['weight_decay']

        save_prefix = dataset_fn + '_' + model_fn
        if self.run_name():
            save_prefix = save_prefix + '_' + self.run_name()

        if self.reporter is None:
            self.reporter = SummaryWriter(log_dir)

        train_data = {
            m: DATASETS[dataset_fn](dataset_root, m) for m in self.modes
        }
        self.prepare_dataset(train_data)
        num_categories = train_data[self.modes[0]].num_categories()

        print('[*] Number of categories:', num_categories)

        net = self.create_model(num_categories)
        net.print_params()

        data_loaders = self.create_data_loaders(train_data)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.params_to_optimize(weight_decay, self.weight_decay_excludes()), lr=learn_rate)
        if learn_rate_step > 0:
            lr_exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=learn_rate_step, gamma=0.5)
        else:
            lr_exp_scheduler = None

        best_accu = 0.0
        best_net = -1

        ckpt_prefix = self.checkpoint_prefix()
        ckpt_nets = self.config['ckpt_nets']
        if ckpt_prefix is not None:
            loaded_paths = net.load(ckpt_prefix, ckpt_nets)
            print('[*] Loaded pretrained model from {}'.format(loaded_paths))

        for epoch in range(1, num_epochs + 1):
            print('-' * 20)
            print('[*] Epoch {}/{}'.format(epoch, num_epochs))

            for mode in self.modes:
                is_train = mode == 'train'
                if not is_train and epoch % valid_freq != 0:
                    continue
                print('[*] Starting {} mode'.format(mode))

                if is_train:
                    if lr_exp_scheduler is not None:
                        lr_exp_scheduler.step()
                    net.train_mode()
                else:
                    net.eval_mode()

                running_corrects = 0
                num_samples = 0
                pbar = tqdm.tqdm(total=len(data_loaders[mode]))
                for bid, data_batch in enumerate(data_loaders[mode]):
                    self.step_counters[mode] += 1

                    logits, loss, gt_category = self.forward_batch(net, data_batch, mode, optimizer, criterion)
                    _, predicts = torch.max(logits, 1)
                    predicts_accu = torch.sum(predicts == gt_category)
                    running_corrects += predicts_accu.item()

                    sampled_batch_size = gt_category.size(0)
                    num_samples += sampled_batch_size

                    if report_scalar_freq > 0 and self.step_counters[mode] % report_scalar_freq == 0:
                        self.reporter.add_scalar('{}/loss'.format(mode), loss.item(), self.step_counters[mode])
                        self.reporter.add_scalar('{}/accuracy'.format(mode),
                                                 float(predicts_accu.data) / sampled_batch_size, self.step_counters[mode])

                    if is_train and save_step_freq > 0 and self.step_counters[mode] % save_step_freq == 0:
                        net.save(log_dir, self.step_counters[mode], save_prefix)
                    pbar.update()
                pbar.close()
                epoch_accu = float(running_corrects) / float(num_samples)

                if is_train:
                    if epoch % save_epoch_freq == 0:
                        print('[*]  {} accu: {:.4f}'.format(mode, epoch_accu))
                        net.save(log_dir, 'epoch_{}'.format(epoch), save_prefix)
                else:
                    print('[*]  {} accu: {:.4f}'.format(mode, epoch_accu))
                    if epoch_accu > best_accu:
                        best_accu = epoch_accu
                        best_net = epoch
        print('[*] Best accu: {:.4f}, corresponding epoch: {}'.format(best_accu, best_net))

        for m in self.modes:
            train_data[m].dispose()

        return best_accu


class SketchR2CNNTrain(BaseTrain):

    def add_args(self, arg_parser):
        arg_parser.add_argument('--dropout', type=float, default=0.5)
        arg_parser.add_argument('--intensity_channels', type=int, default=1)
        return arg_parser

    def create_data_loaders(self, dataset_dict):
        data_loaders = {
            m: DataLoader(dataset_dict[m],
                          batch_size=self.config['batch_size'],
                          num_workers=3 if m == 'train' else 1,
                          shuffle=True if m == 'train' else False,
                          drop_last=True,
                          collate_fn=train_data_collate,
                          pin_memory=True) for m in self.modes
        }
        return data_loaders

    def create_model(self, num_categories):
        dropout = self.config['dropout']
        imgsize = self.config['imgsize']
        intensity_channels = self.config['intensity_channels']
        model_fn = self.config['model_fn']
        thickness = self.config['thickness']

        return SketchR2CNN(CNN_MODELS[model_fn],
                           3,
                           dropout,
                           imgsize,
                           thickness,
                           num_categories,
                           intensity_channels=intensity_channels,
                           device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        imgsize = self.config['imgsize']
        is_train = mode == 'train'
        report_hist_freq = self.config['report_hist_freq']
        report_image_freq = self.config['report_image_freq']
        thickness = self.config['thickness']

        points = data_batch['points3'].to(self.device)
        points_offset = data_batch['points3_offset'].to(self.device)
        points_length = data_batch['points3_length']
        category = data_batch['category'].to(self.device)

        if report_image_freq > 0 and self.step_counters[mode] % report_image_freq == 0:
            images = Raster.to_image(points, 1.0, imgsize, thickness, device=self.device)
            image_grid = torchvision.utils.make_grid(images, nrow=4)
            self.reporter.add_image('{}/sketch_input'.format(mode), image_grid, self.step_counters[mode])

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, attention, images = model(points, points_offset, points_length)
            loss = criterion(logits, category)
            if is_train:
                loss.backward()
                optimizer.step()

        if is_train and report_hist_freq > 0 and self.step_counters[mode] % report_hist_freq == 0:
            self.reporter.add_histogram('{}/attention'.format(mode),
                                        attention,
                                        self.step_counters[mode],
                                        bins='auto')
            self.reporter.add_histogram('{}/points_length'.format(mode),
                                        points_length,
                                        self.step_counters[mode],
                                        bins='auto')
        return logits, loss, category
