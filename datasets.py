import numpy as np
import os
import pandas as pd
import pickle
from random import sample, choices
import torch
from torchvision import transforms
import time
from torch.utils.data import Dataset
from typing import List
from utils import _get_tiles, _choose_data, chunks, map_original_grid_list_to_equiv_grid_list
from transformations import define_transformations
from utils import assert_dataset_target
from utils import get_label
from utils_MIL import dataset_properties_to_location
from utils import balance_dataset
from Dataset_Maker.dataset_utils import get_datasets_dir_dict
from utils import get_optimal_slide_level, cohort_to_int
import openslide
from tqdm import tqdm
import sys
from PIL import Image
from glob import glob
import logging
import cv2


class WSI_Master_Dataset(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 test_time_augmentation: bool = False,
                 desired_slide_magnification: int = 10,
                 slide_repetitions: int = 1,
                 loan: bool = False,
                 er_eq_pr: bool = False,
                 slide_per_block: bool = False,
                 balanced_dataset: bool = False,
                 RAM_saver: bool = False,
                 patch_dir: str = ''
                 ):

        # Multi target training
        N_targets = len(target_kind.split('+'))
        if N_targets > 1:
            target_kind = target_kind.split('+')
            self.multi_target = True
        else:
            self.multi_target = False

        # Check if the target receptor is available for the requested train DataSet:
        assert_dataset_target(DataSet, target_kind)

        logging.info('Initializing {} DataSet....'.format('Train' if train else 'Test'))
        self.DataSet = DataSet
        self.desired_magnification = desired_slide_magnification
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = 'test' if test_fold == 0 else test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.DX = DX
        self.get_images = get_images
        self.train_type = train_type
        self.color_param = color_param
        self.loan = loan
        self.patch_dir = patch_dir

        # Get DataSets location:
        self.dir_dict = get_datasets_dir_dict(Dataset=self.DataSet)
        logging.info('Slide Data will be taken from these locations:')
        logging.info(self.dir_dict)
        locations_list = []

        # allow multiples in TMA dataset to use Amir dataset
        use_multiples = False

        if not self.multi_target:
            if self.target_kind[:3] == 'MRD':
                self.target_kind = 'MRD'

        for _, key in enumerate(self.dir_dict):
            locations_list.append(self.dir_dict[key])

            slide_meta_data_file = os.path.join(self.dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(self.dir_dict[key], 'Grids_' + str(self.desired_magnification),
                                               'Grid_data.xlsx')

            slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
            grid_meta_data_DF = pd.read_excel(grid_meta_data_file)

            if use_multiples:
                if any(grid_meta_data_DF['file'] != slide_meta_data_DF['file']):
                    raise IOError('order of slides in grid file does not match order in slides_data file!')
                grid_meta_data_DF = grid_meta_data_DF.drop('file', axis=1)
                meta_data_DF = pd.concat([slide_meta_data_DF, grid_meta_data_DF], axis=1)  # allow multiples
            else:
                meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
                                             **grid_meta_data_DF.set_index('file').to_dict()})

            self.meta_data_DF = meta_data_DF if not hasattr(self, 'meta_data_DF') else self.meta_data_DF.append(
                meta_data_DF)

        if self.meta_data_DF['id'].isnull().sum() > 0:
            logging.info('Disregarding slides without id')
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'].notnull()]

        if not use_multiples:
            self.meta_data_DF.reset_index(inplace=True)
            self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)

        if self.DataSet == 'PORTO_HE' or self.DataSet == 'PORTO_PDL1':
            # for lung, take only origin: lung
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF.reset_index(inplace=True)

        if balanced_dataset and self.target_kind in ['ER', 'ER100']:
            self.meta_data_DF = balance_dataset(self.meta_data_DF)
            # take only selected patients
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['use_in_balanced_data_ER'] == 1]
            self.meta_data_DF.reset_index(inplace=True)

        if self.target_kind == 'OR':
            PR_targets = list(self.meta_data_DF['PR status'])
            ER_targets = list(self.meta_data_DF['ER status'])
            all_targets = ['Missing Data'] * len(ER_targets)
            for ii, (PR_target, ER_target) in enumerate(zip(PR_targets, ER_targets)):
                if (PR_target == 'Positive' or ER_target == 'Positive'):
                    all_targets[ii] = 'Positive'
                elif (PR_target == 'Negative' or ER_target == 'Negative'):  # avoid 'Missing Data'
                    all_targets[ii] = 'Negative'

        if self.target_kind in ['Survival_Time',
                                'Survival_Binary']:  # TODO: Check if this part can be removed - it might be useless
            all_censored = list(self.meta_data_DF['Censored'])
            all_targets_cont = list(self.meta_data_DF['Time (months)'])
            all_binary_targets = list(self.meta_data_DF['Survival Binary (5 Yr)'])
            all_cohorts = cohort_to_int(list(self.meta_data_DF['id']))

            if self.target_kind == 'Survival_Binary':
                all_targets = all_binary_targets
            elif self.target_kind == 'Survival_Time':
                all_targets = all_targets_cont

        else:
            if self.multi_target:
                all_targets = np.zeros((len(self.meta_data_DF), N_targets), dtype=object)
                for ii in range(N_targets):
                    all_targets[:, ii] = list(self.meta_data_DF[self.target_kind[ii] + ' status'])
            else:
                all_targets = list(self.meta_data_DF[self.target_kind + ' status'])
        all_patient_barcodes = list(self.meta_data_DF['patient barcode'])

        if slide_per_block:
            if DataSet == 'CARMEL':
                all_blocks = []
                for barcode in all_patient_barcodes:
                    if barcode is not np.nan:
                        all_blocks.append(barcode[:-2])
                    else:
                        all_blocks.append(barcode)

                _, unique_inds = np.unique(all_blocks, return_index=True)
                all_inds = np.arange(0, len(all_blocks))
                excess_block_slides = set(all_inds) - set(unique_inds)
                logging.info('slide_per_block: removing ' + str(len(excess_block_slides)) + ' slides')
            else:
                IOError('slide_per_block only implemented for CARMEL dataset')
        elif (DataSet in ['LEUKEMIA', 'ALL']) and (target_kind == 'MRD_day33'):
            excess_block_slides = set(self.meta_data_DF.index[self.meta_data_DF['Day_0/15/33_fixed'] != 33])
        elif (DataSet in ['LEUKEMIA', 'ALL']) and (target_kind == 'MRD_day15'):
            excess_block_slides = set(self.meta_data_DF.index[self.meta_data_DF['Day_0/15/33_fixed'] != 15])
        elif DataSet in ['LEUKEMIA', 'ALL']:  # for leukemia, take only day0 as default
            # remove slides with diagnosis day != 0
            excess_block_slides = set(self.meta_data_DF.index[self.meta_data_DF['Day_0/15/33_fixed'] != 0])
        else:
            excess_block_slides = set()

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        if self.target_kind == 'Survival_Time':
            valid_slide_indices = np.where(np.invert(np.isnan(all_targets)) == True)[0]
        else:
            if self.multi_target:
                valid_slide_indices = \
                    np.where(np.any((all_targets == 'Positive') | (all_targets == 'Negative'), axis=1))[0]
            else:
                all_targets_string = []
                for target in all_targets:
                    if (type(target) == int or type(target) == float) and not np.isnan(target):
                        all_targets_string.append(str(int(target)))
                    else:
                        all_targets_string.append(str(target))

                valid_slide_indices1 = \
                    np.where(np.isin(np.array(all_targets_string), ['Positive', 'Negative']) == True)[0]
                valid_slide_indices2 = np.where(
                    np.isin(np.array(all_targets_string), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']) == True)[
                    0]
                valid_slide_indices = np.hstack((valid_slide_indices1, valid_slide_indices2))

        # inference on unknown labels in case of (blind) test inference or Batched_Full_Slide_Inference_Dataset
        if len(valid_slide_indices) == 0 or self.train_type == 'Infer_All_Folds' or (
                self.target_kind == 'survival' and self.train_type == 'Infer'):
            valid_slide_indices = np.arange(len(all_targets))  # take all slides

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == 0])

        if 'bad segmentation' in self.meta_data_DF.columns:
            slides_with_bad_seg = set(self.meta_data_DF.index[self.meta_data_DF['bad segmentation'] == 1])
        else:
            slides_with_bad_seg = set()

        # train only on samples with ER=PR
        if er_eq_pr and self.train:
            slides_with_er_not_eq_pr = set(
                self.meta_data_DF.index[self.meta_data_DF['ER status'] != self.meta_data_DF['PR status']])
        else:
            slides_with_er_not_eq_pr = set()

        # Define number of tiles to be used
        if train_type == 'REG':
            n_minimal_tiles = n_tiles
        else:
            n_minimal_tiles = self.bag_size

        slides_with_few_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] < n_minimal_tiles])
        # FIXME: find a way to use slides with less than the minimal amount of slides. and than delete the following if.
        if len(slides_with_few_tiles) > 0:
            logging.info(
                '{} Slides were excluded from DataSet because they had less than {} available tiles or are non legitimate for training'
                    .format(len(slides_with_few_tiles), n_minimal_tiles))

        if use_multiples:
            pass  # the "set" kills multiple rows in slides_data
        else:
            valid_slide_indices = np.array(
                list(set(
                    valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles - slides_with_bad_seg - slides_with_er_not_eq_pr - excess_block_slides))

        if RAM_saver:
            # randomly select 1/4 of the slides
            shuffle_factor = 4
            valid_slide_indices = np.random.choice(valid_slide_indices,
                                                   round(len(valid_slide_indices) / shuffle_factor), replace=False)

        # The train set should be a combination of all sets except the test set and validation set:
        if self.DataSet == 'CAT' or self.DataSet == 'ABCTB_TCGA':
            fold_column_name = 'test fold idx breast'
        elif self.target_kind in ['is_tel_aml_B', 'is_tel_aml_non_hr_B']:
            fold_column_name = 'test fold idx for is_tel_aml_B'
        else:
            fold_column_name = 'test fold idx'

        if self.train_type in ['REG', 'MIL']:
            if self.train:
                folds = list(self.meta_data_DF[fold_column_name].unique())
                if test_fold != -1:
                    folds.remove(self.test_fold)
                if 'test' in folds:
                    folds.remove('test')
                if 'val' in folds:
                    folds.remove('val')
            else:
                if test_fold != -1:
                    folds = [self.test_fold, 'val']
                else:
                    folds = []

        elif self.train_type == 'Infer':
            if 0 in infer_folds:
                infer_folds[infer_folds.index(0)] = 'test'
            folds = infer_folds
        elif self.train_type == 'Infer_All_Folds':
            folds = list(self.meta_data_DF[fold_column_name].unique())
        else:
            raise ValueError('Variable train_type is not defined')

        self.folds = folds

        if type(folds) is int:
            folds = [folds]

        correct_folds = self.meta_data_DF[fold_column_name][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_ids = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
            self.desired_magnification)])

        if 'TCGA' not in self.dir_dict:
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Manipulated Objective Power'])

        if train_type in ['Infer', 'Infer_All_Folds']:
            temp_select = False
            if temp_select:  # hard coded selection of specific slides
                slidenames = ['19-14722_1_1_a.mrxs', '19-14722_1_1_b.mrxs', '19-14722_1_1_e.mrxs', '19-5229_2_1_a.mrxs',
                              '19-5229_2_1_b.mrxs', '19-5229_2_1_e.mrxs']
                valid_slide_indices = []
                for slidename in slidenames:
                    valid_slide_index = self.meta_data_DF[self.meta_data_DF['file'] == slidename].index.to_list()
                    valid_slide_indices.append(valid_slide_index[0])

            if self.patch_dir != '':
                # load patches position from excel file
                xfile = glob(os.path.join(self.patch_dir, '*_x.csv'))
                yfile = glob(os.path.join(self.patch_dir, '*_y.csv'))
                if len(xfile) == 0 or len(yfile) == 0:
                    raise IOError('patch location files not found in dir!')
                elif len(xfile) > 1 or len(yfile) > 1:
                    raise IOError('more than one patch location file in dir!')
                self.x_pd = pd.read_csv(xfile[0])
                self.y_pd = pd.read_csv(yfile[0])

                # take all slides that were used in previous inference
                valid_slide_indices_prev = np.array(
                    [int(np.where(self.meta_data_DF['file'] == slide_name)[0]) for slide_name in
                     self.x_pd['slide_name']]).astype('int64')
                valid_slide_indices = valid_slide_indices_prev

            self.valid_slide_indices = valid_slide_indices
            self.all_magnifications = all_magnifications
            self.all_is_DX_cut = all_is_DX_cut if self.DX else [True] * len(self.all_magnifications)
            self.all_tissue_tiles = all_tissue_tiles
            self.all_image_file_names = all_image_file_names
            self.all_patient_barcodes = all_patient_barcodes
            self.all_image_ids = all_image_ids
            self.all_targets = all_targets

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.slides = []
        self.grid_lists = []
        self.presaved_tiles = []

        if self.target_kind in ['Survival_Time', 'Survival_Binary']:
            self.target_binary, self.target_cont, self.censored, self.cohort = [], [], [], []

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(self.dir_dict[all_image_ids[index]])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])
                self.presaved_tiles.append(all_image_ids[index] == 'ABCTB_TILES')

                if self.target_kind in ['Survival_Time', 'Survival_Binary']:
                    self.censored.append(all_censored[index])
                    self.target_binary.append(all_binary_targets[index])
                    self.target_cont.append(all_targets_cont[index])
                    self.cohort.append(all_cohorts[index])

                # Preload slides - improves speed during training.
                grid_file = []
                image_file = []
                try:
                    image_file = os.path.join(self.dir_dict[all_image_ids[index]], all_image_file_names[index])
                    if self.presaved_tiles[-1]:
                        tiles_dir = os.path.join(self.dir_dict[all_image_ids[index]], 'tiles',
                                                 '.'.join((os.path.basename(image_file)).split('.')[:-1]))
                        self.slides.append(tiles_dir)
                        self.grid_lists.append(0)
                    else:
                        if (self.DataSet[:3] == 'TMA'):  # read TMA using opencv
                            self.slides.append(image_file)
                        else:
                            if self.train_type in ['Infer_All_Folds', 'Infer']:
                                self.slides.append(image_file)
                            else:
                                self.slides.append(openslide.open_slide(image_file))

                        basic_file_name = '.'.join(all_image_file_names[index].split('.')[:-1])

                        grid_file = os.path.join(self.dir_dict[all_image_ids[index]],
                                                 'Grids_' + str(self.desired_magnification),
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            if use_multiples:
                                grid_list = [(0, 408)]  # override empty grids - use slides defined by list
                            self.grid_lists.append(grid_list)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Couldn\'t open slide {} or its Grid file {}'.format(image_file, grid_file))

        # Setting the transformation:
        if self.DataSet[:3] == 'TMA':
            norm_type = 'Amir'  # opencv reverses colors, so normalization needs to be reveresed as well
        else:
            norm_type = 'Ron'
        self.transform = define_transformations(transform_type, self.train, self.tile_size, self.color_param, norm_type)
        if np.sum(self.presaved_tiles):
            self.rand_crop = transforms.RandomCrop(self.tile_size)

        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train is False and test_time_augmentation:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)

        # Deleting attributes not needed.
        if train_type == 'Infer_All_Folds':
            attributes_to_delete = ['image_file_names', 'image_path_names', 'slides', 'presaved_tiles']
            for attribute in attributes_to_delete:
                delattr(self, attribute)

        self.random_shift = True if (self.train and self.DataSet[:3] != 'TMA') else False

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        if self.presaved_tiles[idx]:  # load presaved patches
            time_tile_extraction = time.time()
            idxs = sample(range(self.tissue_tiles[idx]), self.bag_size)
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * self.bag_size
            for ii, tile_ind in enumerate(idxs):
                tile_path = os.path.join(self.slides[idx], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
                time_tile_extraction = (time.time() - time_tile_extraction) / len(idxs)
                time_list = [0, time_tile_extraction]
        else:
            slide = self.slides[idx]

            if (self.DataSet[:3] == 'TMA'):  # read TMA using opencv
                im = cv2.imread(self.slides[idx])
                w_margin = (im.shape[1] - 1440) // 2
                im = im[:, w_margin:-w_margin]
                im = cv2.resize(im, dsize=(512, 512),
                                interpolation=cv2.INTER_LINEAR)
                im = Image.fromarray(im)
                tiles = [im]
            else:
                tiles, time_list, label, _ = _choose_data(grid_list=self.grid_lists[idx],
                                                          slide=slide,
                                                          how_many=self.bag_size,
                                                          magnification=self.magnification[idx],
                                                          tile_size=self.tile_size,
                                                          print_timing=self.print_time,
                                                          desired_mag=self.desired_magnification,
                                                          loan=self.loan,
                                                          random_shift=self.random_shift)
        if not self.loan:
            if self.target_kind == 'Survival_Time':
                label = [self.target[idx]]
                label = torch.FloatTensor(label)
            else:
                label = get_label(self.target[idx], self.multi_target)
                label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = self.transform(tiles[i])

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.Compose([transforms.CenterCrop(self.tile_size), transforms.ToTensor()])
            for i in range(self.bag_size):
                images[i] = trans(tiles[i])
        else:
            images = 0

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem

        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
            time_dict = {'Average time to extract a tile': time_list[1],
                         'Augmentation time': aug_time,
                         'Total time': total_time
                         }
        else:
            time_list = [0]
            time_dict = {'Average time to extract a tile': 0,
                         'Augmentation time': 0,
                         'Total time': 0
                         }

        # picking the right value for Target Binary:
        if self.target_kind in ['Survival_Time', 'Survival_Binary']:
            if self.target_binary[idx] == 'Positive':
                target_binary = [1]
            elif self.target_binary[idx] == 'Negative':
                target_binary = [0]
            else:
                target_binary = [-1]
        else:
            target_binary = [-1]

        target_binary = torch.LongTensor(target_binary)

        return {'Data': X,
                'Target': label,
                'Time List': time_list,
                'Time dict': time_dict,
                'File Names': self.image_file_names[idx],
                'Images': images,
                'Target Binary': target_binary,
                'Survival Time': self.target_cont[idx] if self.target_kind in ['Survival_Time',
                                                                               'Survival_Binary'] else torch.LongTensor(
                    [-1]),
                'Censored': bool(self.censored[idx]) if self.target_kind in ['Survival_Time',
                                                                             'Survival_Binary'] else torch.LongTensor(
                    [-1]),
                'Cohort': self.cohort[idx] if self.target_kind in ['Survival_Time',
                                                                   'Survival_Binary'] else torch.LongTensor([-1]),
                'is_Train': self.train
                }


class WSI_REGdataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 10,
                 loan: bool = False,
                 er_eq_pr: bool = False,
                 slide_per_block: bool = False,
                 balanced_dataset: bool = False,
                 RAM_saver: bool = False
                 ):
        super(WSI_REGdataset, self).__init__(DataSet=DataSet,
                                             tile_size=tile_size,
                                             bag_size=1,
                                             target_kind=target_kind,
                                             test_fold=test_fold,
                                             train=train,
                                             print_timing=print_timing,
                                             transform_type=transform_type,
                                             DX=DX,
                                             get_images=get_images,
                                             train_type='REG',
                                             color_param=color_param,
                                             n_tiles=n_tiles,
                                             desired_slide_magnification=desired_slide_magnification,
                                             er_eq_pr=er_eq_pr,
                                             slide_per_block=slide_per_block,
                                             balanced_dataset=balanced_dataset,
                                             RAM_saver=RAM_saver)

        self.loan = loan
        logging.info(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
                .format(self.train_type,
                        'Train' if self.train else 'Test',
                        self.DataSet,
                        self.target_kind,
                        self.desired_magnification,
                        self.real_length,
                        self.tile_size,
                        self.bag_size,
                        'Without' if transform_type == 'none' else 'With',
                        self.test_fold,
                        'ON' if self.DX else 'OFF'))

    def __getitem__(self, idx):
        # X, target, time_list, image_file_names, images = super(WSI_REGdataset, self).__getitem__(idx=idx)
        data_dict = super(WSI_REGdataset, self).__getitem__(idx=idx)
        X = data_dict['Data']
        X = torch.reshape(X, (3, self.tile_size, self.tile_size))

        return {'Data': X,
                'Target': data_dict['Target'],
                'Censored': data_dict['Censored'],
                'Target Binary': data_dict['Target Binary'],
                'Survival Time': data_dict['Survival Time'],
                'Time List': data_dict['Time List'],
                'Time dict': data_dict['Time dict'],
                'File Names': data_dict['File Names'],
                'Images': data_dict['Images'],
                'Cohort': data_dict['Cohort'],
                'is_Train': data_dict['is_Train']
                }


class Infer_Dataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 num_tiles: int = 500,
                 dx: bool = False,
                 desired_slide_magnification: int = 10,
                 resume_slide: int = 0,
                 patch_dir: str = ''
                 ):
        super(Infer_Dataset, self).__init__(DataSet=DataSet,
                                            tile_size=tile_size,
                                            bag_size=None,
                                            target_kind=target_kind,
                                            test_fold=1,
                                            infer_folds=folds,
                                            train=True,
                                            print_timing=False,
                                            transform_type='none',
                                            DX=dx,
                                            get_images=False,
                                            train_type='Infer',
                                            desired_slide_magnification=desired_slide_magnification,
                                            patch_dir=patch_dir)

        self.tiles_per_iter = tiles_per_iter
        self.folds = folds
        self.magnification = []
        self.num_tiles = []
        self.slide_grids = []
        self.grid_lists = []
        self.patient_barcode = []
        self.slide_dataset = []

        ind = 0
        slide_with_not_enough_tiles = 0

        self.valid_slide_indices = self.valid_slide_indices[resume_slide:]
        self.tissue_tiles = self.tissue_tiles[resume_slide:]
        self.image_file_names = self.image_file_names[resume_slide:]
        self.image_path_names = self.image_path_names[resume_slide:]
        self.slides = self.slides[resume_slide:]
        self.presaved_tiles = self.presaved_tiles[resume_slide:]
        self.target = self.target[resume_slide:]

        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= self.all_tissue_tiles[slide_num] and self.all_tissue_tiles[slide_num] > 0:
                    self.num_tiles.append(num_tiles)
                else:
                    self.num_tiles.append(int(self.all_tissue_tiles[slide_num]))
                    slide_with_not_enough_tiles += 1

                self.magnification.extend([self.all_magnifications[slide_num]])
                self.patient_barcode.append(self.all_patient_barcodes[slide_num])
                self.slide_dataset.append(self.all_image_ids[slide_num])
                if self.patch_dir == '':
                    which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])
                else:
                    which_patches = [ii for ii in range(self.num_tiles[-1])]

                patch_ind_chunks = chunks(which_patches, self.tiles_per_iter)
                self.slide_grids.extend(patch_ind_chunks)

                if self.presaved_tiles[ind]:
                    self.grid_lists.append(0)
                else:
                    if self.patch_dir == '':
                        basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                        grid_file = os.path.join(self.image_path_names[ind], 'Grids_' + str(self.desired_magnification),
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            self.grid_lists.append(grid_list)

                ind += 1

        print('There are {} slides with less than {} tiles'.format(slide_with_not_enough_tiles, num_tiles))

        # The following properties will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = -1
        self.current_file = None
        print('Initiation of WSI INFERENCE for {} DataSet and {} of folds {} is Complete'
              .format(self.DataSet,
                      self.target_kind,
                      str(self.folds)))

        print('{} Slides, with X{} magnification. {} tiles per iteration, {} iterations to complete full inference'
              .format(len(self.image_file_names),
                      self.desired_magnification,
                      self.tiles_per_iter,
                      self.__len__()))
        
    def __len__(self):
        return int(np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.slide_num += 1
            self.tiles_to_go = self.num_tiles[self.slide_num]
            self.slide_name = self.image_file_names[self.slide_num]
            self.current_slide = openslide.open_slide(self.slides[self.slide_num])

            self.initial_num_patches = self.num_tiles[self.slide_num]

            if not self.presaved_tiles[self.slide_num]:
                self.best_slide_level, self.adjusted_tile_size, self.level_0_tile_size = \
                    get_optimal_slide_level(self.current_slide, self.magnification[self.slide_num],
                                            self.desired_magnification, self.tile_size)

        label = get_label(self.target[self.slide_num], self.multi_target)
        label = torch.LongTensor(label)

        if self.presaved_tiles[self.slide_num]:
            idxs = self.slide_grids[idx]
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * len(idxs)
            for ii, tile_ind in enumerate(idxs):
                tile_path = os.path.join(self.slides[self.slide_num], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
        else:
            if self.patch_dir == '':
                locs = [self.grid_lists[self.slide_num][loc] for loc in self.slide_grids[idx]]
            else:
                x_loc = [int(self.x_pd.loc[self.x_pd['slide_name'] == self.slide_name][str(loc)].item()) for loc in
                         self.slide_grids[idx]]
                y_loc = [int(self.y_pd.loc[self.y_pd['slide_name'] == self.slide_name][str(loc)].item()) for loc in
                         self.slide_grids[idx]]
                locs = [ii for ii in zip(x_loc, y_loc)]
            tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                             # locations=self.slide_grids[idx],
                                             locations=locs,
                                             tile_size_level_0=self.level_0_tile_size,
                                             adjusted_tile_sz=self.adjusted_tile_size,
                                             output_tile_sz=self.tile_size,
                                             best_slide_level=self.best_slide_level,
                                             random_shift=False)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
        else:
            self.tiles_to_go -= self.tiles_per_iter

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = self.transform(tiles[i])

        aug_time = time.time() - start_aug
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]
        if self.tiles_to_go is None:
            last_batch = True
        else:
            last_batch = False

        return {'Data': X,
                'Label': label,
                'Time List': time_list,
                'Is Last Batch': last_batch,
                'Initial Num Tiles': self.initial_num_patches,
                'Slide Filename': self.slide_name,
                'Patient barcode': self.patient_barcode[self.slide_num],
                'Slide DataSet': self.slide_dataset[self.slide_num],
                'Patch Loc': locs,
                }
    
    def reset_counter(self):
        self.slide_num = -1
        

class Features_MILdataset(Dataset):
    def __init__(self,
                 dataset: str = r'TCGA_ABCTB',
                 data_location: str = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/',
                 bag_size: int = 100,
                 minimum_tiles_in_slide: int = 50,
                 is_per_patient: bool = False,
                 # if True than data will be gathered and returned per patient (else per slide)
                 is_all_tiles: bool = False,
                 # if True than all slide tiles will be used (else only bag_tiles number of tiles)
                 fixed_tile_num: int = None,
                 is_repeating_tiles: bool = True,
                 # if True than the tile pick from each slide/patient is with repeating elements
                 target: str = 'ER',
                 is_train: bool = False,
                 data_limit: int = None,
                 # if 'None' than there will be no data limit. If a number is specified than it'll be the data limit
                 test_fold: int = None,
                 carmel_only: bool = False,
                 # if true than only feature slides from CARMEL dataset will be taken from all the features given the files
                 print_timing: bool = False
                 # slide_repetitions: int = 1
                 ):

        self.is_per_patient, self.is_all_tiles, self.is_repeating_tiles = is_per_patient, is_all_tiles, is_repeating_tiles
        self.bag_size = bag_size
        self.train_type = 'Features'

        target = target.replace('_Features', '', 1) if len(target.split('_')) in [2, 3] and target.split('_')[
            -1] == 'Features' else target  # target.split('_')[0] if len(target.split('_')) == 2 and target.split('_')[1] == 'Features' else target

        self.slide_names = []
        self.labels = []
        self.targets = []
        self.features = []
        self.num_tiles = []
        self.scores = []
        self.tile_scores = []
        self.tile_location = []
        self.patient_data = {}
        self.bad_patient_list = []
        self.fixed_tile_num = fixed_tile_num  # This instance variable indicates what is the number of fixed tiles to be used. if "None" than all tiles will be used. This feature is used to check the necessity in using more than 500 feature tiles for training
        self.carmel_only = carmel_only
        slides_from_same_patient_with_different_target_values, total_slides, bad_num_of_good_tiles = 0, 0, 0
        slides_with_not_enough_tiles, slides_with_bad_segmentation = 0, 0
        patient_list = []

        self.receptor_plus_is_tumor_dset = False
        if type(data_location) is str:
            data_files = glob(os.path.join(data_location, '*.data'))

        elif type(data_location) is dict:
            data_files = []
            for data_key in data_location.keys():
                data_files.extend(glob(os.path.join(data_location[data_key], '*.data')))

        elif type(data_location) is tuple:
            self.receptor_plus_is_tumor_dset = True
            data_files = {'Receptor': [],
                          'is_Tumor': []
                          }
            data_files['Receptor'].extend(glob(os.path.join(data_location[0], '*.data')))
            data_files['is_Tumor'].extend(glob(os.path.join(data_location[1], '*.data')))

        else:
            data_files = []

        print('Loading data from files in location: {}'.format(data_location))

        if type(data_files) is list:
            corrected_data_file = []
            for data_file in data_files:
                if 'features' in data_file.split('/')[-1]:
                    corrected_data_file.append(data_file)

        elif type(
                data_files) is dict:  # FIXME: It is also possible to check for self.receptor_plus_is_tumor_dset == True
            corrected_data_file = {}
            for key in data_files.keys():
                data_files_list_for_dict = []
                for data_file in data_files[key]:
                    if 'features' in data_file.split('/')[-1]:
                        data_files_list_for_dict.append(data_file)

                data_files_list_for_dict.sort()
                corrected_data_file[key] = data_files_list_for_dict

        data_files = corrected_data_file

        if sys.platform == 'darwin':
            if dataset == 'TCGA_ABCTB':
                if target in ['ER', 'ER_Features'] or (
                        target in ['PR', 'PR_Features', 'Her2', 'Her2_Features'] and test_fold == 1):
                    grid_location_dict = {
                        'TCGA': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/TCGA_Grid_data.xlsx',
                        'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_TIF_Grid_data.xlsx'
                    }
                    slide_data_DF_dict = {'TCGA': pd.read_excel(
                        '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_TCGA.xlsx'),
                        'ABCTB': pd.read_excel(
                            '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_ABCTB_TIF.xlsx')
                    }

            elif dataset == 'ABCTB':
                grid_location_dict = {
                    'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_TIF_Grid_data.xlsx'}
                slide_data_DF_dict = {'ABCTB': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_ABCTB.xlsx')}

            elif dataset in ['CAT', 'CAT with Location']:
                grid_location_dict = {
                    'TCGA': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/TCGA_Grid_data.xlsx',
                    'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_TIF_Grid_data.xlsx',
                    'CARMEL': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/CARMEL_Grid_data.xlsx'
                }
                slide_data_DF_dict = {'TCGA': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_TCGA.xlsx'),
                    'ABCTB': pd.read_excel(
                        '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_ABCTB_TIF.xlsx'),
                    'CARMEL': pd.read_excel(
                        '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx')
                }

            elif dataset == 'CARMEL':
                grid_location_dict = {
                    'CARMEL': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/CARMEL_Grid_data.xlsx'}
                slide_data_DF_dict = {'CARMEL': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx')}

            elif dataset == 'CARMEL 9-11':
                grid_location_dict = {
                    'CARMEL Batch 9-11': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/CARMEL_Grid_data_9_11.xlsx'}
                slide_data_DF_dict = {'CARMEL Batch 9-11': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_9_11.xlsx')}

            elif dataset == 'HAEMEK':
                grid_location_dict = {
                    'HAEMEK': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/HAEMEK_Grid_data.xlsx'}
                slide_data_DF_dict = {'HAEMEK': pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_HAEMEK.xlsx')}

            elif dataset == 'CARMEL_40':
                grid_location_dict = {
                    'CARMEL_40': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/CARMEL_40_Grid_data.xlsx'}
                slides_data_DF_CARMEL = pd.read_excel(
                    '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx')
                slides_data_DF_CARMEL.set_index('file', inplace=True)

            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        elif sys.platform == 'linux':
            if dataset == 'TCGA_ABCTB':
                if target in ['ER', 'ER_Features'] or (
                        target in ['PR', 'PR_Features', 'Her2',
                                   'Her2_Features']):  # target in ['PR', 'PR_Features', 'Her2', 'Her2_Features'] and test_fold == 1):
                    grid_location_dict = {'TCGA': r'/mnt/gipmed_new/Data/Breast/TCGA/Grids_10/Grid_data.xlsx',
                                          'ABCTB': r'/mnt/gipmed_new/Data/Breast/ABCTB_TIF/Grids_10/Grid_data.xlsx'
                                          }
                    slide_data_DF_dict = {
                        'TCGA': pd.read_excel(r'/mnt/gipmed_new/Data/Breast/TCGA/slides_data_TCGA.xlsx'),
                        'ABCTB': pd.read_excel(r'/mnt/gipmed_new/Data/Breast/ABCTB_TIF/slides_data_ABCTB.xlsx')
                    }

            elif dataset == 'ABCTB' and target in ['survival', 'survival_Features']:
                grid_location_dict = {'ABCTB': r'/mnt/gipmed_new/Data/Breast/ABCTB_TIF/Grids_10/Grid_data.xlsx'}
                slide_data_DF_dict = {
                    'ABCTB': pd.read_excel(r'/mnt/gipmed_new/Data/Breast/ABCTB_TIF/slides_data_ABCTB.xlsx')}

            elif dataset in ['CAT', 'CAT with Location']:
                grid_location_dict = {'TCGA': r'/mnt/gipmed_new/Data/Breast/TCGA/Grids_10/Grid_data.xlsx',
                                      'ABCTB': r'/mnt/gipmed_new/Data/Breast/ABCTB_TIF/Grids_10/Grid_data.xlsx',
                                      'CARMEL': r'/home/womer/project/All Data/Ran_Features/Grid_data/CARMEL_Grid_data.xlsx'}

                slide_data_DF_dict = {'TCGA': pd.read_excel(r'/mnt/gipmed_new/Data/Breast/TCGA/slides_data_TCGA.xlsx'),
                                      'ABCTB': pd.read_excel(
                                          r'/mnt/gipmed_new/Data/Breast/ABCTB_TIF/slides_data_ABCTB.xlsx'),
                                      'CARMEL': pd.read_excel(
                                          '/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx')
                                      }

            elif dataset == 'HAEMEK':
                grid_location_dict = {
                    'HAEMEK': r'/mnt/gipmed_new/Data/Breast/Haemek/Batch_1/HAEMEK1/Grids_10/Grid_data.xlsx'}
                slide_data_DF_dict = {'HAEMEK': pd.read_excel(
                    '/mnt/gipmed_new/Data/Breast/Haemek/Batch_1/HAEMEK1/slides_data_HAEMEK1.xlsx')}

            elif dataset == 'CARMEL':
                grid_location_dict = {
                    'CARMEL': r'/home/womer/project/All Data/Ran_Features/Grid_data/CARMEL_Grid_data.xlsx'}
                slide_data_DF_dict = {'CARMEL': pd.read_excel(
                    '/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx')}

            elif dataset == 'CARMEL 9-11':
                grid_location_dict = {
                    'CARMEL Batch 9-11': r'/home/womer/project/All Data/Ran_Features/Grid_data/CARMEL_Grid_data_9_11.xlsx'}
                slide_data_DF_dict = {'CARMEL Batch 9-11': pd.read_excel(
                    '/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_9_11.xlsx')}

            elif dataset == 'CARMEL_40':
                grid_location_dict = {
                    'CARMEL_40': r'/home/womer/project/All Data/Ran_Features/Grid_data/CARMEL_40_Grid_data.xlsx'}
                slides_data_DF_CARMEL = pd.read_excel(
                    '/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx')
                slides_data_DF_CARMEL.set_index('file', inplace=True)

            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        grid_DF = pd.DataFrame()
        slide_data_DF = pd.DataFrame()
        for key in grid_location_dict.keys():
            new_grid_DF = pd.read_excel(grid_location_dict[key])
            grid_DF = pd.concat([grid_DF, new_grid_DF])
            slide_data_DF = pd.concat([slide_data_DF, slide_data_DF_dict[key]])

        grid_DF.set_index('file', inplace=True)
        slide_data_DF.set_index('file', inplace=True)

        if type(data_files) is dict:
            data_files_2 = data_files['is_Tumor']
            data_files = data_files['Receptor']

        for file_idx, file in enumerate(tqdm(data_files)):
            with open(file, 'rb') as filehandle:
                inference_data = pickle.load(filehandle)

            try:
                if len(inference_data) == 6:
                    labels, targets, scores, patch_scores, slide_names, features = inference_data
                    tile_location = np.array([[(np.nan, np.nan)] * patch_scores.shape[1]] * patch_scores.shape[0])
                elif len(inference_data) == 7:
                    labels, targets, scores, patch_scores, slide_names, features, batch_number = inference_data
                    tile_location = np.array([[np.nan, np.nan] * patch_scores.shape[1]] * patch_scores.shape[0])
                elif len(inference_data) == 8:
                    labels, targets, scores, patch_scores, slide_names, features, batch_number, tile_location = inference_data
            except ValueError:
                raise Exception('Debug')

            try:
                num_slides, max_tile_num = features.shape[0], features.shape[2]
            except UnboundLocalError:
                print('File Index: {}, File Name: {}, File content length: {}'.format(file_idx, file,
                                                                                      len(inference_data)))
                print(features.shape)

            for slide_num in range(num_slides):
                # The slide '10-14248_1_1_a.mrxs' in Carmel 9 should be named '20-14248_1_1_a.mrxs':
                if slide_names[slide_num] == '10-14248_1_1_a.mrxs':
                    slide_names[slide_num] = '20-14248_1_1_a.mrxs'
                # ABCTB slides WHOSE EXTENTION IS .NDPI SHOULD BE CHANGED TO .TIF:
                if slide_names[slide_num].split('.')[-1] == 'ndpi':
                    slide_names[slide_num] = '.'.join(slide_names[slide_num].split('.')[:-1] + ['tif'])
                # try:
                # # Skip slides that have a "bad segmentation" marker in GridData.xlsx file
                # if grid_DF.loc[slide_names[slide_num], 'bad segmentation'] == 1:
                #     slides_with_bad_segmentation += 1
                #     continue
                # except ValueError:
                #     raise Exception('Debug')
                # except KeyError:
                #     raise Exception('Debug')

                # skip slides that don't belong to CARMEL dataset in case carmel_only flag is TRUE:
                if slide_names[slide_num].split('.')[
                    -1] != 'mrxs' and self.carmel_only:  # This is not a CARMEL slide and carmel_only flag is TRUE
                    continue

                total_slides += 1
                feature_1 = features[slide_num, :, :, 0]
                nan_indices = np.argwhere(np.isnan(feature_1)).tolist()
                first_nan_index = nan_indices[0][1] if bool(nan_indices) else max_tile_num
                tiles_in_slide = first_nan_index  # check if there are any nan values in feature_1

                # The following lines solves the case for dataset with magnification bigger than 10:
                # column_title = 'Legitimate tiles - 256 compatible @ X10' if len(dataset.split('_')) == 1 else 'Legitimate tiles - 256 compatible @ X' + dataset.split('_')[1]
                if len(dataset.split('_')) > 1 and dataset.split('_')[-1].isnumeric():
                    column_title = 'Legitimate tiles - 256 compatible @ X' + dataset.split('_')[-1]
                else:
                    column_title = 'Legitimate tiles - 256 compatible @ X10'

                try:
                    tiles_in_slide_from_grid_data = int(grid_DF.loc[slide_names[slide_num], column_title])
                except TypeError:
                    raise Exception('Debug')

                if tiles_in_slide_from_grid_data < tiles_in_slide:  # Checking that the number of tiles in Grid_data.xlsx is equall to the one found in the actual data
                    bad_num_of_good_tiles += 1
                    tiles_in_slide = tiles_in_slide_from_grid_data

                if data_limit is not None and is_train and tiles_in_slide > data_limit:  # Limit the number of feature tiles according to argument "data_limit
                    tiles_in_slide = data_limit

                if tiles_in_slide < minimum_tiles_in_slide:  # Checking that the slide has a minimum number of tiles to be useable
                    slides_with_not_enough_tiles += 1
                    continue

                if is_per_patient:
                    # calculate patient id:
                    patient = slide_names[slide_num].split('.')[0]
                    if patient.split('-')[0] == 'TCGA':
                        patient = '-'.join(patient.split('-')[:3])
                    elif slide_names[slide_num].split('.')[-1] == 'mrxs':  # This is a CARMEL slide
                        # patient = slides_data_DF_CARMEL.loc[slide_names[slide_num], 'patient barcode']
                        patient = slide_data_DF.loc[slide_names[slide_num], 'patient barcode']

                    # insert to the "all patient list"
                    patient_list.append(patient)

                    # Check if the patient has already been observed to be with multiple targets.
                    # if so count the slide as bad slide and move on to the next slide
                    if patient in self.bad_patient_list:
                        slides_from_same_patient_with_different_target_values += 1
                        continue

                    # in case this patient has already been seen, than it has multiple slides
                    if patient in self.patient_data.keys():
                        patient_dict = self.patient_data[patient]

                        # Check if the patient has multiple targets
                        patient_same_target = True if int(targets[slide_num]) == patient_dict[
                            'target'] else False  # Checking the the patient target is not changing between slides
                        # if the patient has multiple targets than:
                        if not patient_same_target:
                            slides_from_same_patient_with_different_target_values += 1 + len(patient_dict[
                                                                                                 'slides'])  # we skip more than 1 slide since we need to count the current slide and the ones that are already inserted to the patient_dict
                            self.patient_data.pop(patient)  # remove it from the dictionary of legitimate patients
                            self.bad_patient_list.append(patient)  # insert it to the list of non legitimate patients
                            continue  # and move on to the next slide

                        patient_dict['num tiles'].append(tiles_in_slide)
                        patient_dict['tile scores'] = np.concatenate(
                            (patient_dict['tile scores'], patch_scores[slide_num, :tiles_in_slide]), axis=0)
                        patient_dict['labels'].append(int(labels[slide_num]))
                        # A patient with multiple slides has only 1 target, therefore another target should not be inserted into the dict
                        # patient_dict['target'].append(int(targets[slide_num]))
                        patient_dict['slides'].append(slide_names[slide_num])
                        patient_dict['scores'].append(scores[slide_num])
                        patient_dict['tile locations'].extend(tile_location[slide_num][:tiles_in_slide])

                        # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                        if self.fixed_tile_num is not None:
                            tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                        features_old = patient_dict['features']
                        features_new = features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32)
                        patient_dict['features'] = np.concatenate((features_old, features_new), axis=0)

                    else:
                        patient_dict = {'num tiles': [tiles_in_slide],
                                        'features': features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(
                                            np.float32),
                                        'tile scores': patch_scores[slide_num, :tiles_in_slide],
                                        'labels': [int(labels[slide_num])],
                                        'target': int(targets[slide_num]),
                                        'slides': [slide_names[slide_num]],
                                        'scores': [scores[slide_num]],
                                        'tile locations': tile_location[slide_num][:tiles_in_slide].tolist()
                                        }

                        self.patient_data[patient] = patient_dict

                else:
                    # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                    if self.fixed_tile_num is not None:
                        tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                    self.num_tiles.append(tiles_in_slide)
                    self.features.append(features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32))
                    if len(patch_scores.shape) == 2:
                        self.tile_scores.append(patch_scores[slide_num, :tiles_in_slide])
                    elif len(patch_scores.shape) == 1:
                        self.tile_scores.append(patch_scores[:tiles_in_slide])

                    self.slide_names.append(slide_names[slide_num])
                    self.labels.append(int(labels[slide_num]))
                    self.targets.append(int(targets[slide_num]))
                    self.scores.append(scores[slide_num])
                    # self.tile_location.append(tile_location[slide_num, :tiles_in_slide, :])
                    self.tile_location.append(tile_location[slide_num, :tiles_in_slide])

                # Checking for consistency between targets loaded from the feature files and slides_data_DF.
                # The location of this check should include per patient dataset or per slide dataset
                if dataset != 'CARMEL 9-11':
                    target_for_PD = target.split('+')[0] if '+' in target else target
                    slide_data_target = slide_data_DF.loc[slide_names[slide_num]][target_for_PD + ' status']
                    if slide_data_target == 'Positive':
                        slide_data_target = 1
                    elif slide_data_target == 'Negative':
                        slide_data_target = 0
                    else:
                        slide_data_target = -1

                    feature_file_target = targets[slide_num]
                    if slide_data_target != feature_file_target:
                        raise Exception('Found inconsistency between targets in feature files and slide_data_DF')

        # Organizing the receptor slide data in a dict for the case where we want datasets that contain Receptor + is_Tumor features:
        if self.receptor_plus_is_tumor_dset and not self.is_per_patient:  # We need to arrange only the data per slide (the PER PATIENT data is already ordered in a dict)
            self.Receptor_slide_data = {'slide_names': self.slide_names,
                                        'labels': self.labels,
                                        'targets': self.targets,
                                        'scores': self.scores,
                                        'tile_scores': self.tile_scores,
                                        'features': self.features,
                                        'num_tiles': self.num_tiles,
                                        'tile_location': self.tile_location
                                        }

        if 'data_files_2' in locals():
            self.is_Tumor_slide_data = {'num_tiles': [],
                                        'features': [],
                                        'slide_names': [],
                                        'tile_location': []
                                        }

            slides_from_same_patient_with_different_target_values_is_Tumor, total_slides_is_Tumor, bad_num_of_good_tiles_is_Tumor = 0, 0, 0
            slides_with_not_enough_tiles_is_Tumor, slides_with_bad_segmentation_is_Tumor = 0, 0

            if self.is_per_patient:
                self.patient_data_is_Tumor = {}

            for file_idx, file in enumerate(tqdm(data_files_2)):
                with open(file, 'rb') as filehandle:
                    inference_data = pickle.load(filehandle)

                try:
                    if len(inference_data) == 6:
                        labels, targets, scores, patch_scores, slide_names, features = inference_data
                        tile_location = np.array([[(np.nan, np.nan)] * patch_scores.shape[1]] * patch_scores.shape[0])
                    elif len(inference_data) == 7:
                        labels, targets, scores, patch_scores, slide_names, features, batch_number = inference_data
                        tile_location = np.array([[np.nan, np.nan] * patch_scores.shape[1]] * patch_scores.shape[0])
                    elif len(inference_data) == 8:
                        labels, targets, scores, patch_scores, slide_names, features, batch_number, tile_location = inference_data
                except ValueError:
                    raise Exception('Debug')

                try:
                    num_slides, max_tile_num = features.shape[0], features.shape[2]
                except UnboundLocalError:
                    print(file_idx, file, len(inference_data))
                    print(features.shape)

                for slide_num in range(num_slides):
                    # The slide '10-14248_1_1_a.mrxs' in Carmel 9 should be named '20-14248_1_1_a.mrxs':
                    if slide_names[slide_num] == '10-14248_1_1_a.mrxs':
                        slide_names[slide_num] = '20-14248_1_1_a.mrxs'
                    # ABCTB slides WHOSE EXTENTION IS .NDPI SHOULD BE CHANGED TO .TIF:
                    if slide_names[slide_num].split('.')[-1] == 'ndpi':
                        slide_names[slide_num] = '.'.join(slide_names[slide_num].split('.')[:-1] + ['tif'])
                    try:
                        # Skip slides that have a "bad segmentation" marker in GridData.xlsx file
                        if grid_DF.loc[slide_names[slide_num], 'bad segmentation'] == 1:
                            slides_with_bad_segmentation_is_Tumor += 1
                            continue
                    except ValueError:
                        raise Exception('Debug')
                    except KeyError:
                        raise Exception('Debug')

                    # skip slides that don't belong to CARMEL dataset in case carmel_only flag is TRUE:
                    if slide_names[slide_num].split('.')[
                        -1] != 'mrxs' and self.carmel_only:  # This is not a CARMEL slide and carmel_only flag is TRUE
                        continue

                    total_slides_is_Tumor += 1
                    feature_1 = features[slide_num, :, :, 0]
                    nan_indices = np.argwhere(np.isnan(feature_1)).tolist()
                    tiles_in_slide = nan_indices[0][1] if bool(
                        nan_indices) else max_tile_num  # check if there are any nan values in feature_1

                    # The following lines solves the case for dataset with magnification bigger than 10:
                    # column_title = 'Legitimate tiles - 256 compatible @ X10' if len(dataset.split('_')) == 1 else 'Legitimate tiles - 256 compatible @ X' + dataset.split('_')[1]
                    if len(dataset.split('_')) > 1 and dataset.split('_')[-1].isnumeric():
                        column_title = 'Legitimate tiles - 256 compatible @ X' + dataset.split('_')[-1]
                    else:
                        column_title = 'Legitimate tiles - 256 compatible @ X10'

                    try:
                        tiles_in_slide_from_grid_data = int(grid_DF.loc[slide_names[slide_num], column_title])
                    except TypeError:
                        raise Exception('Debug')

                    if tiles_in_slide_from_grid_data < tiles_in_slide:  # Checking that the number of tiles in Grid_data.xlsx is equall to the one found in the actual data
                        bad_num_of_good_tiles_is_Tumor += 1
                        tiles_in_slide = tiles_in_slide_from_grid_data

                    if data_limit is not None and is_train and tiles_in_slide > data_limit:  # Limit the number of feature tiles according to argument "data_limit
                        tiles_in_slide = data_limit

                    if tiles_in_slide < minimum_tiles_in_slide:  # Checking that the slide has a minimum number of tiles to be useable
                        slides_with_not_enough_tiles_is_Tumor += 1
                        continue

                    if is_per_patient:
                        # calculate patient id:
                        patient = slide_names[slide_num].split('.')[0]
                        if patient.split('-')[0] == 'TCGA':
                            patient = '-'.join(patient.split('-')[:3])
                        elif slide_names[slide_num].split('.')[-1] == 'mrxs':  # This is a CARMEL slide
                            # patient = slides_data_DF_CARMEL.loc[slide_names[slide_num], 'patient barcode']
                            patient = slide_data_DF.loc[slide_names[slide_num], 'patient barcode']

                        # insert to the "all patient list"
                        patient_list.append(patient)

                        ### This check is not relevant for is_Tumor !
                        '''# Check if the patient has already been observed to be with multiple targets.
                        # if so count the slide as bad slide and move on to the next slide                        
                        if patient in self.bad_patient_list_is_Tumor:
                            slides_from_same_patient_with_different_target_values_is_Tumor += 1
                            continue'''

                        # in case this patient has already been seen, than it has multiple slides
                        if patient in self.patient_data_is_Tumor.keys():
                            patient_dict = self.patient_data_is_Tumor[patient]

                            # Check if the patient has multiple targets - This check is not relevant for is_Tumor
                            '''patient_same_target = True if int(targets[slide_num]) == patient_dict['target'] else False  # Checking the the patient target is not changing between slides
                            # if the patient has multiple targets than:
                            if not patient_same_target:
                                slides_from_same_patient_with_different_target_values_is_Tumor += 1 + len(patient_dict['slides'])  # we skip more than 1 slide since we need to count the current slide and the ones that are already inserted to the patient_dict
                                self.patient_data_is_Tumor.pop(patient)  # remove it from the dictionary of legitimate patients
                                self.bad_patient_list_is_Tumor.append(patient)  # insert it to the list of non legitimate patients
                                continue  # and move on to the next slide'''

                            patient_dict['num tiles'].append(tiles_in_slide)
                            patient_dict['tile scores'] = np.concatenate(
                                (patient_dict['tile scores'], patch_scores[slide_num, :tiles_in_slide]), axis=0)
                            patient_dict['labels'].append(int(labels[slide_num]))
                            # A patient with multiple slides has only 1 target, therefore another target should not be inserted into the dict
                            # patient_dict['target'].append(int(targets[slide_num]))
                            patient_dict['slides'].append(slide_names[slide_num])
                            patient_dict['scores'].append(scores[slide_num])
                            patient_dict['tile locations'].extend(tile_location[slide_num][:tiles_in_slide])

                            # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                            if self.fixed_tile_num is not None:
                                tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                            features_old = patient_dict['features']
                            features_new = features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32)
                            patient_dict['features'] = np.concatenate((features_old, features_new), axis=0)

                        else:
                            patient_dict = {'num tiles': [tiles_in_slide],
                                            'features': features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(
                                                np.float32),
                                            'tile scores': patch_scores[slide_num, :tiles_in_slide],
                                            'labels': [int(labels[slide_num])],
                                            'target': int(targets[slide_num]),
                                            'slides': [slide_names[slide_num]],
                                            'scores': [scores[slide_num]],
                                            'tile locations': tile_location[slide_num][:tiles_in_slide].tolist()
                                            }

                            self.patient_data_is_Tumor[patient] = patient_dict

                    else:
                        # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                        if self.fixed_tile_num is not None:
                            tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                        self.is_Tumor_slide_data['num_tiles'].append(tiles_in_slide)
                        self.is_Tumor_slide_data['features'].append(
                            features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32))

                        self.is_Tumor_slide_data['slide_names'].append(slide_names[slide_num])
                        self.is_Tumor_slide_data['tile_location'].append(tile_location[slide_num, :tiles_in_slide])

            # Organizing 2 datasets as one:
            if self.is_per_patient:
                self.dSet = self.__organize_dsets__(receptor_dset=self.patient_data,
                                                    is_tumor_dset=self.patient_data_is_Tumor,
                                                    is_per_patient=True)
            else:
                self.dSet = self.__organize_dsets__(receptor_dset=self.Receptor_slide_data,
                                                    is_tumor_dset=self.is_Tumor_slide_data,
                                                    is_per_patient=False)

        print(
            'There are {}/{} slides whose tile amount in Grid_data.xlsx is lower than amount found in the feature files'.format(
                bad_num_of_good_tiles, total_slides))
        print('There are {}/{} slides with \"bad segmentation\" '.format(slides_with_bad_segmentation, total_slides))
        print('There are {}/{} slides with less than {} tiles '.format(slides_with_not_enough_tiles, total_slides,
                                                                       minimum_tiles_in_slide))

        if self.is_per_patient:
            self.patient_keys = list(self.patient_data.keys())
            self.patient_set = set(patient_list)
            print('Skipped {}/{} slides for {}/{} patients (Inconsistent target value for same patient)'.format(
                slides_from_same_patient_with_different_target_values, total_slides, len(self.bad_patient_list),
                len(set(patient_list))))
            print('Initialized Dataset with {} feature slides in {} patients'.format(
                total_slides - slides_from_same_patient_with_different_target_values - slides_with_not_enough_tiles,
                self.__len__()))
        else:
            print('Initialized Dataset with {} feature slides'.format(self.__len__()))

    def __len__(self):
        if self.is_per_patient:
            return len(self.patient_keys)
        else:
            return len(self.slide_names)

    def __organize_dsets__(self, receptor_dset, is_tumor_dset, is_per_patient):
        if is_per_patient:  # datasets are per patient
            all_patient_data = {}
            for patient_id in receptor_dset.keys():
                if receptor_dset[patient_id]['num tiles'] != is_tumor_dset[patient_id]['num tiles']:
                    raise Exception('num tiles number for patient are not equal')
                if receptor_dset[patient_id]['slides'] != is_tumor_dset[patient_id]['slides']:
                    raise Exception('slide names for patient are not equal')
                if not (np.array(receptor_dset[patient_id]['tile locations']) == np.array(
                        is_tumor_dset[patient_id]['tile locations'])).all():
                    raise Exception('tile locations for patient are not equal')

                # Gather all data in one dataset:
                all_patient_data[patient_id] = {'slide_names': receptor_dset[patient_id]['slides'],
                                                'labels': receptor_dset[patient_id]['labels'],
                                                'targets': receptor_dset[patient_id]['target'],
                                                'features': receptor_dset[patient_id]['features'],
                                                'tumor_features': is_tumor_dset[patient_id]['features'],
                                                'num_tiles': receptor_dset[patient_id]['num tiles'],
                                                'scores': receptor_dset[patient_id]['scores'],
                                                'tile_scores': receptor_dset[patient_id]['tile scores'],
                                                'tile_location': receptor_dset[patient_id]['tile locations']
                                                }

            return all_patient_data

        else:  # datasets are per slide
            slide_data = {'slide_names': [],
                          'labels': [],
                          'targets': [],
                          'features': [],
                          'tumor_features': [],
                          'num_tiles': [],
                          'scores': [],
                          'tile_scores': [],
                          'tile_location': []
                          }

            if not receptor_dset['num_tiles'] == is_tumor_dset['num_tiles']:
                raise Exception('num_tiles of the 2 datasets are NOT equivalent')

            for idx, slide_name in enumerate(receptor_dset['slide_names']):
                if not (receptor_dset['tile_location'][idx] == is_tumor_dset['tile_location'][idx]).all():
                    raise Exception('The tile locations of the 2 datasets are NOT equivalent')

                slide_data['slide_names'].append(slide_name)
                slide_data['tile_location'].append(receptor_dset['tile_location'][idx])
                slide_data['labels'].append(receptor_dset['labels'][idx])
                slide_data['targets'].append(receptor_dset['targets'][idx])
                slide_data['num_tiles'].append(receptor_dset['num_tiles'][idx])
                slide_data['scores'].append(receptor_dset['scores'][idx])
                slide_data['tile_scores'].append(receptor_dset['tile_scores'][idx])
                slide_data['features'].append(receptor_dset['features'][idx])
                slide_data['tumor_features'].append(is_tumor_dset['features'][idx])

            return slide_data

    def __getitem__(self, item):
        if self.receptor_plus_is_tumor_dset:
            if self.is_per_patient:
                patient_data = self.dSet[self.patient_keys[item]]
                num_tiles = int(np.array(patient_data['num_tiles']).sum())

                if self.is_repeating_tiles:
                    tile_idx = list(range(num_tiles)) if self.is_all_tiles else choices(range(num_tiles),
                                                                                        k=self.bag_size)
                else:
                    tile_idx = list(range(num_tiles)) if self.is_all_tiles else sample(range(num_tiles),
                                                                                       k=self.bag_size)

                return {'labels': np.array(patient_data['labels']).mean(),
                        'targets': patient_data['targets'],
                        'scores': np.array(patient_data['scores']).mean(),
                        'tile scores': patient_data['tile_scores'][tile_idx],
                        'features': patient_data['features'][tile_idx],
                        'tumor_features': patient_data['tumor_features'][tile_idx],
                        'num tiles': num_tiles,
                        'tile locations': np.array(patient_data['tile_location'])[tile_idx]
                        }

            else:
                tile_idx = list(range(self.dSet['num_tiles'][item])) if self.is_all_tiles else choices(
                    range(self.dSet['num_tiles'][item]), k=self.bag_size)

                return {'labels': self.dSet['labels'][item],
                        'targets': self.dSet['targets'][item],
                        'scores': self.dSet['scores'][item],
                        'tile scores': self.dSet['tile_scores'][item][tile_idx],
                        'slide name': self.dSet['slide_names'][item],
                        'features': self.dSet['features'][item][tile_idx],
                        'tumor_features': self.dSet['tumor_features'][item][tile_idx],
                        'num tiles': self.dSet['num_tiles'][item],
                        'tile locations': self.dSet['tile_location'][item][tile_idx]
                        }

        else:
            if self.is_per_patient:
                patient_data = self.patient_data[self.patient_keys[item]]
                num_tiles = int(np.array(patient_data['num tiles']).sum())

                if self.is_repeating_tiles:
                    tile_idx = list(range(num_tiles)) if self.is_all_tiles else choices(range(num_tiles),
                                                                                        k=self.bag_size)
                else:
                    tile_idx = list(range(num_tiles)) if self.is_all_tiles else sample(range(num_tiles),
                                                                                       k=self.bag_size)

                return {'labels': np.array(patient_data['labels']).mean(),
                        'targets': patient_data['target'],
                        'scores': np.array(patient_data['scores']).mean(),
                        'tile scores': patient_data['tile scores'][tile_idx],
                        'features': patient_data['features'][tile_idx],
                        'num tiles': num_tiles
                        }

            else:
                tile_idx = list(range(self.num_tiles[item])) if self.is_all_tiles else choices(
                    range(self.num_tiles[item]), k=self.bag_size)

                return {'labels': self.labels[item],
                        'targets': self.targets[item],
                        'scores': self.scores[item],
                        'tile scores': self.tile_scores[item][tile_idx],
                        'slide name': self.slide_names[item],
                        'features': self.features[item][tile_idx],
                        'num tiles': self.num_tiles[item],
                        'tile locations': self.tile_location[item][tile_idx] if hasattr(self, 'tile_location') else None
                        }


class Combined_Features_for_MIL_Training_dataset(Dataset):
    def __init__(self,
                 dataset_list: list = ['CAT', 'CARMEL'],  # for Multi_Resolution [CARMEL_10, CARMEL_40]
                 similar_dataset: str = 'CARMEL',
                 bag_size: int = 100,
                 minimum_tiles_in_slide: int = 50,
                 is_per_patient: bool = False,
                 # if True than data will be gathered and returned per patient (else per slide)
                 is_all_tiles: bool = False,
                 # if True than all slide tiles will be used (else only bag_tiles number of tiles)
                 fixed_tile_num: int = None,
                 is_repeating_tiles: bool = True,
                 # if True than the tile pick from each slide/patient is with repeating elements
                 target: str = 'ER',
                 is_train: bool = False,
                 data_limit: int = None,
                 # if 'None' than there will be no data limit. If a number is specified than it'll be the data limit
                 test_fold: int = 1,
                 print_timing: bool = False,
                 slide_repetitions: int = 1
                 ):

        if similar_dataset != 'CARMEL':
            raise Exception('ONLY the case were CARMEL is the similar_dataset is implemented')

        self.is_per_patient, self.is_all_tiles, self.is_repeating_tiles = is_per_patient, is_all_tiles, is_repeating_tiles
        self.dataset_list = dataset_list
        self.bag_size = bag_size
        self.train_type = 'Features'
        self.fixed_tile_num = fixed_tile_num  # This instance variable indicates what is the number of fixed tiles to be used. if "None" than all tiles will be used. This feature is used to check the necessity in using more than 500 feature tiles for training

        if self.is_per_patient:
            self.patient_data = {}
        else:
            self.slide_data = {}

        # Get data location:
        datasets_location = dataset_properties_to_location(dataset_list, target, test_fold, is_train)

        bad_num_of_good_tiles, slides_with_not_enough_tiles, slides_with_bad_segmentation = 0, 0, 0
        total_slides = 0
        slides_from_same_patient_with_different_target_values = 0

        print('Gathering data for {} Slides'.format(similar_dataset))
        for dataset, data_location, dataset_name, _ in datasets_location:
            print('Loading data from files related to {} Dataset'.format(dataset))

            if self.is_per_patient:  # Variables needed when working per patient:
                patient_list = []
                bad_patient_list = []
                patient_data = {}
                slides_from_same_patient_with_different_target_values = 0
            else:
                dataset_slide_names = []
                dataset_labels = []
                dataset_targets = []
                dataset_features = []
                dataset_num_tiles = []
                dataset_scores = []
                dataset_tile_scores = []

            data_files = glob(os.path.join(data_location, '*.data'))
            for data_file in data_files:
                if 'features' not in data_file.split('_'):
                    data_files.remove(data_file)

            '''if os.path.join(data_location, 'Model_Epoch_1000-Folds_[2, 3, 4, 5]_ER-Tiles_500.data') in data_files:
                data_files.remove(os.path.join(data_location, 'Model_Epoch_1000-Folds_[2, 3, 4, 5]_ER-Tiles_500.data'))
            if os.path.join(data_location, 'Model_Epoch_1000-Folds_[1]_ER-Tiles_500.data') in data_files:
                data_files.remove(os.path.join(data_location, 'Model_Epoch_1000-Folds_[1]_ER-Tiles_500.data'))'''

            if sys.platform == 'darwin':
                if dataset == 'CAT':
                    grid_location_dict = {
                        # 'TCGA': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/TCGA_Grid_data.xlsx',
                        # 'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_TIF_Grid_data.xlsx',
                        'CARMEL': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/CARMEL_Grid_data.xlsx'}
                    slides_data_DF_CARMEL = pd.read_excel(
                        '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx')
                    slides_data_DF_CARMEL.set_index('file', inplace=True)

                elif dataset == 'CARMEL':
                    grid_location_dict = {
                        'CARMEL': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/CARMEL_Grid_data.xlsx'}
                    slides_data_DF_CARMEL = pd.read_excel(
                        '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx')
                    slides_data_DF_CARMEL.set_index('file', inplace=True)
                elif dataset == 'CARMEL_40':
                    grid_location_dict = {
                        'CARMEL_40': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/CARMEL_40_Grid_data.xlsx'}
                    slides_data_DF_CARMEL = pd.read_excel(
                        '/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx')
                    slides_data_DF_CARMEL.set_index('file', inplace=True)

                else:
                    raise Exception("Need to write which dictionaries to use for this receptor case")

            elif sys.platform == 'linux':
                if dataset == 'CAT':
                    grid_location_dict = {
                        # 'TCGA': r'/mnt/gipmed_new/Data/Breast/TCGA/Grids_10/Grid_data.xlsx',
                        # 'ABCTB': r'/mnt/gipmed_new/Data/ABCTB_TIF/Grids_10/Grid_data.xlsx',
                        'CARMEL': r'/home/womer/project/All Data/Ran_Features/Grid_data/CARMEL_Grid_data.xlsx'}
                    slides_data_DF_CARMEL = pd.read_excel(
                        '/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx')
                    slides_data_DF_CARMEL.set_index('file', inplace=True)

                elif dataset == 'CARMEL':
                    grid_location_dict = {
                        'CARMEL': r'/home/womer/project/All Data/Ran_Features/Grid_data/CARMEL_Grid_data.xlsx'}
                    slides_data_DF_CARMEL = pd.read_excel(
                        '/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx')
                    slides_data_DF_CARMEL.set_index('file', inplace=True)

                elif dataset == 'CARMEL_40':
                    grid_location_dict = {
                        'CARMEL_40': r'/home/womer/project/All Data/Ran_Features/Grid_data/CARMEL_40_Grid_data.xlsx'}
                    slides_data_DF_CARMEL = pd.read_excel(
                        '/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx')
                    slides_data_DF_CARMEL.set_index('file', inplace=True)

                else:
                    raise Exception("Need to write which dictionaries to use in this receptor case")

            grid_DF = pd.DataFrame()
            for key in grid_location_dict.keys():
                new_grid_DF = pd.read_excel(grid_location_dict[key])
                grid_DF = pd.concat([grid_DF, new_grid_DF])

            grid_DF.set_index('file', inplace=True)

            for file_idx, file in enumerate(tqdm(data_files)):
                with open(os.path.join(data_location, file), 'rb') as filehandle:
                    inference_data = pickle.load(filehandle)

                try:
                    if len(inference_data) == 6:
                        labels, targets, scores, patch_scores, slide_names, features = inference_data
                    elif len(inference_data) == 7:
                        labels, targets, scores, patch_scores, slide_names, features, batch_number = inference_data
                except ValueError:
                    raise Exception('Debug')

                num_slides, max_tile_num = features.shape[0], features.shape[2]

                for slide_num in range(num_slides):
                    # skip slides that are not in the similar dataset:
                    if slide_names[slide_num].split('.')[-1] != 'mrxs':  # This is NOT a CARMEL slide
                        continue
                    # Skip slides that have a "bad segmentation" marker in GridData.xlsx file
                    try:
                        if grid_DF.loc[slide_names[slide_num], 'bad segmentation'] == 1:
                            slides_with_bad_segmentation += 1
                            continue
                    except ValueError:
                        raise Exception('Debug')
                    except KeyError:
                        raise Exception('Debug')

                    total_slides += 1
                    feature_1 = features[slide_num, :, :, 0]
                    nan_indices = np.argwhere(np.isnan(feature_1)).tolist()
                    tiles_in_slide = nan_indices[0][1] if bool(
                        nan_indices) else max_tile_num  # check if there are any nan values in feature_1
                    column_title = 'Legitimate tiles - 256 compatible @ X10' if len(
                        dataset.split('_')) == 1 else 'Legitimate tiles - 256 compatible @ X' + dataset.split('_')[1]
                    try:
                        tiles_in_slide_from_grid_data = int(grid_DF.loc[slide_names[slide_num], column_title])
                    except TypeError:
                        raise Exception('Debug')

                    if tiles_in_slide_from_grid_data < tiles_in_slide:  # Checking that the number of tiles in Grid_data.xlsx is equall to the one found in the actual data
                        bad_num_of_good_tiles += 1
                        tiles_in_slide = tiles_in_slide_from_grid_data

                    if data_limit is not None and is_train and tiles_in_slide > data_limit:  # Limit the number of feature tiles according to argument "data_limit
                        tiles_in_slide = data_limit

                    if tiles_in_slide < minimum_tiles_in_slide:  # Checking that the slide has a minimum number of tiles to be useable
                        slides_with_not_enough_tiles += 1
                        continue

                    # Now, we'll start going over the slides and save the data. we're doing that in two loops. One for each dataset.
                    if self.is_per_patient:
                        # Calculate patient id:
                        patient = slide_names[slide_num].split('.')[0]
                        if patient.split('-')[0] == 'TCGA':
                            patient = '-'.join(patient.split('-')[:3])
                        elif slide_names[slide_num].split('.')[-1] == 'mrxs':  # This is a CARMEL slide
                            patient = slides_data_DF_CARMEL.loc[slide_names[slide_num], 'patient barcode']

                        # insert to the "all patient list"
                        patient_list.append(patient)

                        # Check if the patient has already been observed to be with multiple targets.
                        # if so count the slide as bad slide and move on to the next slide
                        if patient in bad_patient_list:
                            slides_from_same_patient_with_different_target_values += 1
                            continue

                        # in case this patient has already been seen, than it has multiple slides
                        if patient in patient_data.keys():
                            patient_dict = patient_data[patient]

                            # Check if the patient has multiple targets
                            patient_same_target = True if int(targets[slide_num]) == patient_dict[
                                'target'] else False  # Checking the the patient target is not changing between slides
                            # if the patient has multiple targets than:
                            if not patient_same_target:
                                slides_from_same_patient_with_different_target_values += 1 + len(patient_dict[
                                                                                                     'slide names'])  # we skip more than 1 slide since we need to count the current slide and the ones that are already inserted to the patient_dict
                                patient_data.pop(patient)  # remove it from the dictionary of legitimate patients
                                bad_patient_list.append(patient)  # insert it to the list of non legitimate patients
                                continue  # and move on to the next slide
                            patient_dict['num tiles'].append(tiles_in_slide)
                            patient_dict['tile scores'].append(patch_scores[slide_num,
                                                               :tiles_in_slide])  # = np.concatenate((patient_dict['tile scores'], patch_scores[slide_num, :tiles_in_slide]), axis=0)
                            patient_dict['labels'].append(int(labels[slide_num]))
                            patient_dict['slide names'].append(slide_names[slide_num])
                            patient_dict['slide scores'].append(scores[slide_num])

                            # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                            if self.fixed_tile_num is not None:
                                tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                            # features_old = patient_dict['features']
                            # features_new = features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32)
                            # patient_dict['features'] = np.concatenate((features_old, features_new), axis=0)
                            patient_dict['features'].append(
                                features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32))

                        else:
                            patient_dict = {'num tiles': [tiles_in_slide],
                                            'features': [features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(
                                                np.float32)],
                                            'tile scores': [patch_scores[slide_num, :tiles_in_slide]],
                                            'labels': [int(labels[slide_num])],
                                            'target': int(targets[slide_num]),
                                            'slide names': [slide_names[slide_num]],
                                            'slide scores': [scores[slide_num]]
                                            }
                            patient_data[patient] = patient_dict

                    else:
                        dataset_num_tiles.append(tiles_in_slide)
                        dataset_features.append(
                            features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32))
                        dataset_tile_scores.append(patch_scores[slide_num, :tiles_in_slide])
                        dataset_slide_names.append(slide_names[slide_num])
                        dataset_labels.append(int(labels[slide_num]))
                        dataset_targets.append(int(targets[slide_num]))
                        dataset_scores.append(scores[slide_num])

            if self.is_per_patient:
                print(
                    'Found {} slides in {} Dataset, from same patients with different targets. Those slides will be excluded from {}'
                        .format(slides_from_same_patient_with_different_target_values,
                                dataset,
                                'Training' if is_train else 'Testing'))

                # self.slide_data[dataset] = patient_data
                self.patient_data[dataset] = patient_data
            else:
                self.slide_data[dataset] = {'slide names': dataset_slide_names,
                                            'features': dataset_features,
                                            'tile scores': dataset_tile_scores,
                                            'labels': dataset_labels,
                                            'targets': dataset_targets,
                                            'slide scores': dataset_scores,
                                            'num tiles': dataset_num_tiles
                                            }

        # At this point we've extracted all the data relevant for the similar dataset from both feature datasets.
        # Now, we'll check that both datasets has the same slides and that they are sorted in the same order.
        # Checking if both datasets has the same slides:
        if self.is_per_patient:
            # Checking that the patient list in each dataset is equal:
            # if set(self.slide_data[datasets_location[0][0]].keys()) - set(self.slide_data[datasets_location[1][0]].keys()):
            if set(self.patient_data[datasets_location[0][0]].keys()) - set(
                    self.patient_data[datasets_location[1][0]].keys()):
                raise Exception('Datasets has different amount of patients. This case is not implemented')
            # Checking that the slide names for each patient is equal:
            # patient_list = self.slide_data[datasets_location[0][0]].keys()
            patient_list = self.patient_data[datasets_location[0][0]].keys()
            for patient in patient_list:
                # if set(self.slide_data[datasets_location[0][0]][patient]['slide names']) - set(self.slide_data[datasets_location[1][0]][patient]['slide names']):
                if set(self.patient_data[datasets_location[0][0]][patient]['slide names']) - set(
                        self.patient_data[datasets_location[1][0]][patient]['slide names']):
                    raise Exception(
                        'Datasets has different amount of slides for patient {}. This case is not implemented'.format(
                            patient))

        else:
            set_1 = set(self.slide_data[datasets_location[0][0]]['slide names'])
            set_2 = set(self.slide_data[datasets_location[1][0]]['slide names'])
            set_1_2 = set_1 - set_2
            set_2_1 = set_2 - set_1
            if bool(set_1_2) or bool(set_2_1):
                print('Datasets has different amount of slides. Removing slides that don\'t reside in both datasets')

                if len(set_1_2) != 0:
                    different_slides_1_2 = list(set_1_2)
                    slide_location_1_2 = []
                    for slide in different_slides_1_2:
                        slide_location_1_2.append(self.slide_data[datasets_location[0][0]]['slide names'].index(slide))

                    # Removing all slide data that do not reside in the other dataset:
                    for slide_location in slide_location_1_2:
                        del self.slide_data[datasets_location[0][0]]['slide names'][slide_location]
                        del self.slide_data[datasets_location[0][0]]['features'][slide_location]
                        del self.slide_data[datasets_location[0][0]]['tile scores'][slide_location]
                        del self.slide_data[datasets_location[0][0]]['labels'][slide_location]
                        del self.slide_data[datasets_location[0][0]]['targets'][slide_location]
                        del self.slide_data[datasets_location[0][0]]['slide scores'][slide_location]
                        del self.slide_data[datasets_location[0][0]]['num tiles'][slide_location]

                    print('Removed slides {} from {} dataset'.format(different_slides_1_2, datasets_location[0][0]))

                if len(set_2_1) != 0:
                    different_slides_2_1 = list(set_2_1)
                    slide_location_2_1 = []
                    for slide in different_slides_2_1:
                        slide_location_2_1.append(self.slide_data[datasets_location[1][0]]['slide names'].index(slide))

                    # Removing all slide data that do not reside in the other dataset:
                    for slide_location in slide_location_2_1:
                        del self.slide_data[datasets_location[1][0]]['slide names'][slide_location]
                        del self.slide_data[datasets_location[1][0]]['features'][slide_location]
                        del self.slide_data[datasets_location[1][0]]['tile scores'][slide_location]
                        del self.slide_data[datasets_location[1][0]]['labels'][slide_location]
                        del self.slide_data[datasets_location[1][0]]['targets'][slide_location]
                        del self.slide_data[datasets_location[1][0]]['slide scores'][slide_location]
                        del self.slide_data[datasets_location[1][0]]['num tiles'][slide_location]

                    print('Removed slides {} from {} dataset'.format(different_slides_2_1, datasets_location[1][0]))

        # Sorting both datasets:
        print('Sorting the Data per {} ...'.format('Patient' if self.is_per_patient else 'Slide'))
        for dataset, _, _, _ in datasets_location:
            if self.is_per_patient:
                # The data in both dictionaries is arranged per patient and can be called by the patient number.
                # We need now to sort the tile data in each patient so it'll be the same for both dictionaries.
                # We already checked that both dataset contain the same patients and each patient has the same slides
                for patient in list(self.patient_data[dataset].keys()):
                    sort_order = np.argsort(np.array(self.patient_data[dataset][patient]['slide names']))
                    self.patient_data[dataset][patient]['slide names'] = [
                        self.patient_data[dataset][patient]['slide names'][index] for index in sort_order]
                    self.patient_data[dataset][patient]['features'] = [
                        self.patient_data[dataset][patient]['features'][index] for index in sort_order]
                    self.patient_data[dataset][patient]['tile scores'] = [
                        self.patient_data[dataset][patient]['tile scores'][index] for index in sort_order]
                    self.patient_data[dataset][patient]['labels'] = [
                        self.patient_data[dataset][patient]['labels'][index] for index in sort_order]
                    self.patient_data[dataset][patient]['slide scores'] = [
                        self.patient_data[dataset][patient]['slide scores'][index] for index in sort_order]
                    self.patient_data[dataset][patient]['num tiles'] = [
                        self.patient_data[dataset][patient]['num tiles'][index] for index in sort_order]

                    '''self.patient_data[dataset][patient]['slide names'] =    list(np.array(self.patient_data[dataset][patient]['slide names'])[sort_order])
                    self.patient_data[dataset][patient]['features'] =       list(np.array(self.patient_data[dataset][patient]['features'], dtype=object)[sort_order])
                    self.patient_data[dataset][patient]['tile scores'] =    list(np.array(self.patient_data[dataset][patient]['tile scores'], dtype=object)[sort_order])
                    self.patient_data[dataset][patient]['labels'] =         list(np.array(self.patient_data[dataset][patient]['labels'])[sort_order])
                    self.patient_data[dataset][patient]['slide scores'] =   list(np.array(self.patient_data[dataset][patient]['slide scores'])[sort_order])
                    self.patient_data[dataset][patient]['num tiles'] =      list(np.array(self.patient_data[dataset][patient]['num tiles'])[sort_order])'''

                    # We'll now convert the Features (self.patient_data[dataset][patient]['features']) and the tile scores into one big array:
                    self.patient_data[dataset][patient]['features'] = np.concatenate(
                        self.patient_data[dataset][patient]['features'], axis=0).astype(np.float32)
                    self.patient_data[dataset][patient]['tile scores'] = np.concatenate(
                        self.patient_data[dataset][patient]['tile scores'], axis=0).astype(np.float32)
            else:
                sort_order = np.argsort(np.array(self.slide_data[dataset]['slide names']))
                self.slide_data[dataset]['slide names'] = [self.slide_data[dataset]['slide names'][index] for index in
                                                           sort_order]
                self.slide_data[dataset]['features'] = [self.slide_data[dataset]['features'][index] for index in
                                                        sort_order]
                self.slide_data[dataset]['tile scores'] = [self.slide_data[dataset]['tile scores'][index] for index in
                                                           sort_order]
                self.slide_data[dataset]['labels'] = [self.slide_data[dataset]['labels'][index] for index in sort_order]
                self.slide_data[dataset]['targets'] = [self.slide_data[dataset]['targets'][index] for index in
                                                       sort_order]
                self.slide_data[dataset]['slide scores'] = [self.slide_data[dataset]['slide scores'][index] for index in
                                                            sort_order]
                self.slide_data[dataset]['num tiles'] = [self.slide_data[dataset]['num tiles'][index] for index in
                                                         sort_order]

                '''self.slide_data[dataset]['slide names'] = list(np.array(self.slide_data[dataset]['slide names'])[sort_order])
                self.slide_data[dataset]['features'] = list(np.array(self.slide_data[dataset]['features'], dtype=object)[sort_order])
                self.slide_data[dataset]['tile scores'] = list(np.array(self.slide_data[dataset]['tile scores'], dtype=object)[sort_order])
                self.slide_data[dataset]['labels'] = list(np.array(self.slide_data[dataset]['labels'])[sort_order])
                self.slide_data[dataset]['targets'] = list(np.array(self.slide_data[dataset]['targets'])[sort_order])
                self.slide_data[dataset]['slide scores'] = list(np.array(self.slide_data[dataset]['slide scores'])[sort_order])
                self.slide_data[dataset]['num tiles'] = list(np.array(self.slide_data[dataset]['num tiles'])[sort_order])'''

        if self.is_per_patient:
            self.patient_list = list(patient_list)
            for patient in self.patient_list:
                # Checking that the targets are equal for both datasets and that there is an equal number of tiles per slide:
                if self.patient_data[datasets_location[0][0]][patient]['target'] - \
                        self.patient_data[datasets_location[1][0]][patient]['target'] != 0:
                    raise Exception('Datasets targets for patient {} are not equal'.format(patient))

        else:
            # Checking that the targets are equal for both datasets and that there is an equal number of tiles per slide:
            if abs(np.array(self.slide_data[datasets_location[0][0]]['targets']) - np.array(
                    self.slide_data[datasets_location[1][0]]['targets'])).sum() != 0:
                raise Exception('Datasets targets are not equal')

            # Checking if the num tiles are equal for both datasets:
            if abs(np.array(self.slide_data[datasets_location[0][0]]['num tiles']) - np.array(
                    self.slide_data[datasets_location[1][0]]['num tiles'])).sum() != 0:
                print('Datasets num tiles are not equal')

        bad_num_of_good_tiles //= len(datasets_location)
        slides_with_bad_segmentation //= len(datasets_location)
        slides_with_not_enough_tiles //= len(datasets_location)
        total_slides //= len(datasets_location)
        if bad_num_of_good_tiles:
            print(
                'There are {}/{} slides with \"bad number of good tile\" '.format(bad_num_of_good_tiles, total_slides))
        if slides_with_bad_segmentation:
            print(
                'There are {}/{} slides with \"bad segmentation\" '.format(slides_with_bad_segmentation, total_slides))
        if slides_with_not_enough_tiles:
            print('There are {}/{} slides with less than {} tiles '.format(slides_with_not_enough_tiles, total_slides,
                                                                           minimum_tiles_in_slide))

        print('Initialized Dataset with {} feature slides'.format(self.__len__()))

    def __len__(self):
        if self.is_per_patient:
            return len(self.patient_list)
        else:
            return len(self.slide_data[list(self.slide_data.keys())[0]]['slide names'])

    def __getitem__(self, item):
        if self.is_per_patient:  # Retrieving data per patient
            patient = self.patient_list[item]
            dataset_names = list(self.patient_data.keys())

            patient_data = self.patient_data[self.dataset_list[0]][patient]
            num_tiles = int(np.array(patient_data['num tiles']).sum())

            if self.is_repeating_tiles:
                tile_idx = list(range(num_tiles)) if self.is_all_tiles else choices(range(num_tiles), k=self.bag_size)
            else:
                tile_idx = list(range(num_tiles)) if self.is_all_tiles else sample(range(num_tiles), k=self.bag_size)

            return {dataset_names[0]:
                        {'targets': self.patient_data[dataset_names[0]][patient]['target'],
                         'tile scores': self.patient_data[dataset_names[0]][patient]['tile scores'][tile_idx],
                         'features': self.patient_data[dataset_names[0]][patient]['features'][tile_idx],
                         'num tiles': int(np.array(self.patient_data[dataset_names[0]][patient]['num tiles']).sum())
                         },
                    dataset_names[1]:
                        {'targets': self.patient_data[dataset_names[1]][patient]['target'],
                         'tile scores': self.patient_data[dataset_names[1]][patient]['tile scores'][tile_idx],
                         'features': self.patient_data[dataset_names[1]][patient]['features'][tile_idx],
                         'num tiles': int(np.array(self.patient_data[dataset_names[1]][patient]['num tiles']).sum())
                         }
                    }

        else:  # Retrieving data per slide
            tile_idx = list(range(
                self.slide_data[list(self.slide_data.keys())[0]]['num tiles'][item])) if self.is_all_tiles else choices(
                range(self.slide_data[list(self.slide_data.keys())[0]]['num tiles'][item]), k=self.bag_size)
            dataset_names = list(self.slide_data.keys())

            return {dataset_names[0]:
                        {'targets': self.slide_data[dataset_names[0]]['targets'][item],
                         'slide scores': self.slide_data[dataset_names[0]]['slide scores'][item],
                         'tile scores': self.slide_data[dataset_names[0]]['tile scores'][item][tile_idx],
                         'slide name': self.slide_data[dataset_names[0]]['slide names'][item],
                         'features': self.slide_data[dataset_names[0]]['features'][item][tile_idx],
                         'num tiles': self.slide_data[dataset_names[0]]['num tiles'][item]
                         },
                    dataset_names[1]:
                        {'targets': self.slide_data[dataset_names[1]]['targets'][item],
                         'slide scores': self.slide_data[dataset_names[1]]['slide scores'][item],
                         'tile scores': self.slide_data[dataset_names[1]]['tile scores'][item][tile_idx],
                         'slide name': self.slide_data[dataset_names[1]]['slide names'][item],
                         'features': self.slide_data[dataset_names[1]]['features'][item][tile_idx],
                         'num tiles': self.slide_data[dataset_names[1]]['num tiles'][item]
                         }
                    }


class One_Full_Slide_Inference_Dataset(WSI_Master_Dataset):
    """
    This Class provides tile extraction for ONE specific slide WITHOUT using the legitimate tile grid
    """

    def __init__(self,
                 DataSet: str = 'TCGA',
                 slidename: str = '',
                 target_kind: str = 'ER',
                 folds: List = [1],
                 tile_size: int = 256,
                 desired_slide_magnification: int = 10
                 ):
        super(One_Full_Slide_Inference_Dataset, self).__init__(DataSet=DataSet,
                                                               tile_size=256,
                                                               bag_size=None,
                                                               target_kind=target_kind,
                                                               test_fold=1,
                                                               infer_folds=folds,
                                                               train=True,
                                                               print_timing=False,
                                                               transform_type='none',
                                                               get_images=False,
                                                               train_type='Infer',
                                                               desired_slide_magnification=desired_slide_magnification)

        self.tile_size = tile_size

        slide_idx = np.where(np.array(self.all_image_file_names) == slidename)[0][0]

        height = int(self.meta_data_DF.loc[slide_idx, 'Height'])
        width = int(self.meta_data_DF.loc[slide_idx, 'Width'])
        objective_power = self.meta_data_DF.loc[slide_idx, 'Manipulated Objective Power']

        adjusted_tile_size_at_level_0 = int(self.tile_size * (int(objective_power) / self.desired_magnification))
        equivalent_rows = int(np.ceil(height / adjusted_tile_size_at_level_0))
        equivalent_cols = int(np.ceil(width / adjusted_tile_size_at_level_0))

        self.delta_pixel = int(objective_power / self.desired_magnification)

        self.equivalent_grid_size = (equivalent_rows, equivalent_cols)
        self.magnification = self.all_magnifications[slide_idx]
        self.slide_name = self.all_image_file_names[slide_idx]
        self.slide = self.slides[slide_idx]

    def __len__(self):
        return 1

    def __getitem__(self, location: List = None):
        '''desired_downsample = self.magnification / self.desired_magnification

        level, best_next_level = -1, -1
        for index, downsample in enumerate(self.slide.level_downsamples):
            if isclose(desired_downsample, downsample, rel_tol=1e-3):
                level = index
                level_downsample = 1
                break

            elif downsample < desired_downsample:
                best_next_level = index
                level_downsample = int(
                    desired_downsample / self.slide.level_downsamples[best_next_level])

        self.adjusted_tile_size = self.tile_size * level_downsample

        self.best_slide_level = level if level > best_next_level else best_next_level
        self.level_0_tile_size = int(desired_downsample) * self.tile_size'''

        self.best_slide_level, self.adjusted_tile_size, self.level_0_tile_size = \
            get_optimal_slide_level(self.slide, self.magnification, self.desired_magnification, self.tile_size)

        tiles, time_list, _ = _get_tiles(slide=self.slide,
                                         locations=location,
                                         tile_size_level_0=self.level_0_tile_size,
                                         adjusted_tile_sz=self.adjusted_tile_size,
                                         output_tile_sz=self.tile_size,
                                         best_slide_level=self.best_slide_level)

        # Converting the tiles to Tensor:
        X, original_data = [], []
        for tile in tiles:
            original_data.append(transforms.ToTensor()(tile))
            X.append(torch.reshape(self.transform(tile), (
                1, self.transform(tile).size(0), self.transform(tile).size(1), self.transform(tile).size(2))))

        return {'Data': X,
                'Slide Filename': self.slide._filename,
                'Equivalent Grid Size': self.equivalent_grid_size,
                'Original Data': original_data
                }


class Batched_Full_Slide_Inference_Dataset(WSI_Master_Dataset):
    def __init__(self,
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 desired_slide_magnification: int = 10,
                 num_background_tiles: int = 0
                 ):
        f = open('Infer_Slides.txt', 'r')
        DataSet = f.readline().split('\n')[0]
        self.slide_names = []
        for line in f:
            if line.isspace():
                continue
            self.slide_names.append(line.split('\n')[0])
        f.close()

        super(Batched_Full_Slide_Inference_Dataset, self).__init__(DataSet=DataSet,
                                                                   tile_size=tile_size,
                                                                   bag_size=None,
                                                                   target_kind=target_kind,
                                                                   train=True,
                                                                   print_timing=False,
                                                                   transform_type='none',
                                                                   DX=False,
                                                                   get_images=False,
                                                                   train_type='Infer_All_Folds',
                                                                   desired_slide_magnification=desired_slide_magnification)

        self.tiles_per_iter = tiles_per_iter
        self.magnification = []
        self.num_tiles = []
        self.slide_grids = []
        self.equivalent_grid = []
        self.equivalent_grid_size = []
        self.is_tissue_tiles = []
        self.is_last_batch = []
        self.slides = []
        targets, slide_size = [], []

        for _, slide_num in enumerate(self.valid_slide_indices):
            if self.all_image_file_names[slide_num] not in self.slide_names:
                continue
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                try:
                    # slides.append(openslide.open_slide(self.slides[slide_num]))

                    full_slide_file_name = os.path.join(self.dir_dict[self.all_image_ids[slide_num]],
                                                        self.all_image_file_names[slide_num])
                    print(self.all_image_file_names[slide_num])
                    self.slides.append(openslide.open_slide(full_slide_file_name))
                except FileNotFoundError:
                    # raise FileNotFoundError('Couldn\'t open slide {}'.format(self.slides[slide_num]))
                    raise FileNotFoundError('Couldn\'t open slide {}'.format(full_slide_file_name))

                targets.append(self.all_targets[slide_num])

                # Recreate the basic slide grids:
                height = int(self.meta_data_DF.loc[slide_num, 'Height'])
                width = int(self.meta_data_DF.loc[slide_num, 'Width'])
                slide_size.append({'Height': height, 'Width': width})
                objective_power = self.meta_data_DF.loc[slide_num, 'Manipulated Objective Power']

                adjusted_tile_size_at_level_0 = int(
                    self.tile_size * (int(objective_power) / self.desired_magnification))
                equivalent_rows = int(np.ceil(height / adjusted_tile_size_at_level_0))
                equivalent_cols = int(np.ceil(width / adjusted_tile_size_at_level_0))
                basic_grid = [(row, col) for row in range(0, height, adjusted_tile_size_at_level_0) for col in
                              range(0, width, adjusted_tile_size_at_level_0)]
                equivalent_grid_dimensions = (equivalent_rows, equivalent_cols)
                self.equivalent_grid_size.append(equivalent_grid_dimensions)

                if len(basic_grid) != self.meta_data_DF.loc[
                    slide_num, 'Total tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
                        self.desired_magnification)].item():
                    raise Exception('Total tile num do not fit')

                self.magnification.extend([self.all_magnifications[slide_num]])
                basic_file_name = '.'.join(self.all_image_file_names[slide_num].split('.')[:-1])
                grid_file = os.path.join(self.dir_dict[self.all_image_ids[slide_num]],
                                         'Grids_' + str(self.desired_magnification),
                                         basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                '''grid_file = os.path.join(self.image_path_names[ind], 'Grids_' + str(self.desired_magnification),
                                         basic_file_name + '--tlsz' + str(self.tile_size) + '.data')'''

                with open(grid_file, 'rb') as filehandle:
                    tissue_grid_list = pickle.load(filehandle)
                # Compute which tiles are background tiles and pick "num_background_tiles" from them.
                non_tissue_grid_list = list(set(basic_grid) - set(tissue_grid_list))
                selected_non_tissue_grid_list = sample(non_tissue_grid_list, num_background_tiles)
                # Combine the list of selected non tissue tiles with the list of all tiles:
                combined_grid_list = selected_non_tissue_grid_list + tissue_grid_list
                combined_equivalent_grid_list = map_original_grid_list_to_equiv_grid_list(adjusted_tile_size_at_level_0,
                                                                                          combined_grid_list)
                # We'll also create a list that says which tiles are tissue tiles:
                is_tissue_tile = [False] * len(selected_non_tissue_grid_list) + [True] * len(tissue_grid_list)

                self.num_tiles.append(len(combined_grid_list))

                chosen_locations_chunks = chunks(combined_grid_list, self.tiles_per_iter)
                self.slide_grids.extend(chosen_locations_chunks)

                chosen_locations_equivalent_grid_chunks = chunks(combined_equivalent_grid_list, self.tiles_per_iter)
                self.equivalent_grid.extend(chosen_locations_equivalent_grid_chunks)

                is_tissue_tile_chunks = chunks(is_tissue_tile, self.tiles_per_iter)
                self.is_tissue_tiles.extend(is_tissue_tile_chunks)
                self.is_last_batch.extend([False] * (len(chosen_locations_chunks) - 1))
                self.is_last_batch.append(True)

                print('Slide: {}, num tiles: {}, num batches: {}'
                      .format(self.slides[-1], self.num_tiles[-1], len(chosen_locations_chunks)))

        self.targets = targets
        self.slide_size = slide_size

        attributes_to_delete = ['all_image_file_names', 'all_is_DX_cut', 'all_magnifications',
                                'all_patient_barcodes', 'all_tissue_tiles', 'grid_lists', 'in_fold',
                                'target', 'tissue_tiles', 'valid_slide_indices']
        for attribute in attributes_to_delete:
            delattr(self, attribute)
        # The following properties will be used in the __getitem__ function
        self.slide_num = -1
        self.current_file = None
        print(
            'Initiation of WSI Batch Slides for {} INFERENCE is Complete. {} Slides, Working on Tiles of size {}^2 with X{} magnification. {} tiles per iteration, {} iterations to complete full inference'
                .format(self.target_kind,
                        len(self.slides),
                        self.tile_size,
                        self.desired_magnification,
                        self.tiles_per_iter,
                        self.__len__()))

    def __len__(self):
        return int(np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    # def __getitem__(self, idx):
    def __getitem__(self, idx, location: List = None, tile_size: int = None):
        start_getitem = time.time()
        if idx == 0 or (idx > 0 and self.is_last_batch[idx - 1]):
            self.slide_num += 1

            if sys.platform == 'win32':
                image_file = os.path.join(self.image_path_names[self.slide_num],
                                          self.image_file_names[self.slide_num])
                self.current_slide = openslide.open_slide(image_file)
            else:
                self.current_slide = self.slides[self.slide_num]

            self.initial_num_patches = self.num_tiles[self.slide_num]

            if tile_size is not None:
                self.tile_size = tile_size

            '''desired_downsample = self.magnification[self.slide_num] / self.desired_magnification

            level, best_next_level = -1, -1
            for index, downsample in enumerate(self.current_slide.level_downsamples):
                if isclose(desired_downsample, downsample, rel_tol=1e-3):
                    level = index
                    level_downsample = 1
                    break

                elif downsample < desired_downsample:
                    best_next_level = index
                    level_downsample = int(
                        desired_downsample / self.current_slide.level_downsamples[best_next_level])

            self.adjusted_tile_size = self.tile_size * level_downsample

            self.best_slide_level = level if level > best_next_level else best_next_level
            self.level_0_tile_size = int(desired_downsample) * self.tile_size'''

            self.best_slide_level, self.adjusted_tile_size, self.level_0_tile_size = \
                get_optimal_slide_level(self.current_slide, self.magnification[self.slide_num],
                                        self.desired_magnification, self.tile_size)

        target = get_label(self.all_targets[self.slide_num], self.multi_target)
        target = torch.LongTensor(target)

        tile_locations = self.slide_grids[idx] if location is None else location

        tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                         locations=tile_locations,
                                         tile_size_level_0=self.level_0_tile_size,
                                         adjusted_tile_sz=self.adjusted_tile_size,
                                         output_tile_sz=self.tile_size,
                                         best_slide_level=self.best_slide_level)

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])
        tiles_non_augmented = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = self.transform(tiles[i])
            tiles_non_augmented[i] = transforms.ToTensor()(tiles[i])

        aug_time = time.time() - start_aug
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        return {'Data': X,
                'Target': target,
                'Time List': time_list,
                'Is Last Batch': self.is_last_batch[idx],
                'Initial Num Tiles': self.initial_num_patches,
                'Slide Filename': self.current_slide._filename.split('/')[-1],
                'Equivalent Grid': self.equivalent_grid[idx],
                'Is Tissue Tiles': self.is_tissue_tiles[idx],
                'Equivalent Grid Size': self.equivalent_grid_size[self.slide_num],
                'Level 0 Locations': self.slide_grids[idx],
                'Original Data': tiles_non_augmented,
                'Slide Dimensions': self.slide_size[self.slide_num]
                }


class WSI_Master_Dataset_Survival_CR(Dataset):
    def __init__(self,
                 DataSet: str = 'ABCTB',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'survival',
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 transform_type: str = 'none',
                 DX: bool = False,
                 get_images: bool = False,
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 10,
                 Censored_ratio: float = 0.5,
                 legit_slide_list: list = []):

        # Multi target training
        N_targets = len(target_kind.split('+'))
        if N_targets > 1:
            target_kind = target_kind.split('+')
            self.multi_target = True
        else:
            self.multi_target = False

        # Check if the target receptor is available for the requested train DataSet:
        assert_dataset_target(DataSet, target_kind)

        print('Initializing {} DataSet....'.format('Train' if train else 'Test'))
        self.DataSet = DataSet
        self.desired_magnification = desired_slide_magnification
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.DX = DX
        self.get_images = get_images
        self.train_type = train_type
        self.color_param = color_param
        self.censored_ratio = Censored_ratio

        # Get DataSets location:
        self.dir_dict = get_datasets_dir_dict(Dataset=self.DataSet)
        print('Slide Data will be taken from these locations:')
        print(self.dir_dict)
        locations_list = []

        for _, key in enumerate(self.dir_dict):
            locations_list.append(self.dir_dict[key])

            slide_meta_data_file = os.path.join(self.dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(self.dir_dict[key], 'Grids_' + str(self.desired_magnification),
                                               'Grid_data.xlsx')

            slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
            grid_meta_data_DF = pd.read_excel(grid_meta_data_file)
            meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
                                         **grid_meta_data_DF.set_index('file').to_dict()})

            self.meta_data_DF = meta_data_DF if not hasattr(self, 'meta_data_DF') else self.meta_data_DF.append(
                meta_data_DF)

        self.meta_data_DF.reset_index(inplace=True)
        self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)

        if self.DataSet == 'PORTO_HE' or self.DataSet == 'PORTO_PDL1':
            # for lung, take only origin: lung
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF.reset_index(inplace=True)

        if self.target_kind == 'OR':
            PR_targets = list(self.meta_data_DF['PR status'])
            ER_targets = list(self.meta_data_DF['ER status'])
            all_targets = ['Missing Data'] * len(ER_targets)
            for ii, (PR_target, ER_target) in enumerate(zip(PR_targets, ER_targets)):
                if PR_target == 'Positive' or ER_target == 'Positive':
                    all_targets[ii] = 'Positive'
                elif PR_target == 'Negative' or ER_target == 'Negative':  # avoid 'Missing Data'
                    all_targets[ii] = 'Negative'

        if self.target_kind == 'survival':
            all_censored = list(self.meta_data_DF['Censored'])
            all_time_targets = list(self.meta_data_DF['Follow-up Months Since Diagnosis'])
            all_binary_targets = list(self.meta_data_DF['survival status'])
            all_is_excluded = list(self.meta_data_DF['Exclude for time prediction?'])

        else:
            all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        all_patient_barcodes = list(self.meta_data_DF['patient barcode'])

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        if self.target_kind == 'survival':
            # In the case where the target is survival we need to exclude slides that are:
            # 1) Excluded
            # 2) without Censor data

            # valid_slide_indices_exclude = np.where(np.array(all_is_excluded) != 'Exclude')[0]
            valid_slide_indices_exclude = np.where(np.array(all_is_excluded) != 'Exclude')[0]
            valid_slide_indices_censor = np.where(np.isnan(all_censored) == False)[0]
            # Combining all those conditions:
            valid_slide_indices = np.intersect1d(valid_slide_indices_exclude, valid_slide_indices_censor)

        else:
            valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == 0])

        if 'bad segmentation' in self.meta_data_DF.columns:
            slides_with_bad_seg = set(self.meta_data_DF.index[self.meta_data_DF['bad segmentation'] == 1])
        else:
            slides_with_bad_seg = set()

        # Define number of tiles to be used
        if train_type == 'REG':
            n_minimal_tiles = n_tiles
        else:
            n_minimal_tiles = self.bag_size

        slides_with_few_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] < n_minimal_tiles])
        # FIXME: find a way to use slides with less than the minimal amount of slides. and than delete the following if.
        if len(slides_with_few_tiles) > 0:
            print(
                '{} Slides were excluded from DataSet because they had less than {} available tiles or are non legitimate for training'
                    .format(len(slides_with_few_tiles), n_minimal_tiles))
        valid_slide_indices = np.array(
            list(set(
                valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles - slides_with_bad_seg))

        # The train set should be a combination of all sets except the test set and validation set:
        if self.DataSet == 'CAT' or self.DataSet == 'ABCTB_TCGA':
            fold_column_name = 'test fold idx breast'
        else:
            fold_column_name = 'test fold idx'

        if self.train_type in ['REG', 'MIL']:
            if self.train:
                folds = list(self.meta_data_DF[fold_column_name].unique())
                if self.test_fold in folds:  # if the dataset was balanced than the test fold might not be in it.
                    folds.remove(self.test_fold)
                if 'test' in folds:
                    folds.remove('test')
                if 'val' in folds:
                    folds.remove('val')
            else:
                folds = [self.test_fold]
                folds.append('val')
        elif self.train_type == 'Infer':
            folds = infer_folds
        elif self.train_type == 'Infer_All_Folds':
            folds = list(self.meta_data_DF[fold_column_name].unique())
        else:
            raise ValueError('Variable train_type is not defined')

        self.folds = folds

        if type(folds) is int:
            folds = [folds]

        correct_folds = self.meta_data_DF[fold_column_name][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_ids = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
            self.desired_magnification)])

        if 'TCGA' not in self.dir_dict:
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Manipulated Objective Power'])

        if train_type in ['Infer', 'Infer_All_Folds']:
            temp_select = False
            if temp_select or len(legit_slide_list) > 0:  # hard coded selection of specific slides
                if temp_select:
                    slidenames = ['19-14722_1_1_a.mrxs', '19-14722_1_1_b.mrxs', '19-14722_1_1_e.mrxs',
                                  '19-5229_2_1_a.mrxs',
                                  '19-5229_2_1_b.mrxs', '19-5229_2_1_e.mrxs']
                else:
                    slidenames = legit_slide_list

                valid_slide_indices = []
                for slidename in slidenames:
                    valid_slide_index = self.meta_data_DF[self.meta_data_DF['file'] == slidename].index.to_list()
                    valid_slide_indices.append(valid_slide_index[0])

            self.valid_slide_indices = valid_slide_indices
            self.all_magnifications = all_magnifications
            self.all_is_DX_cut = all_is_DX_cut if self.DX else [True] * len(self.all_magnifications)
            self.all_tissue_tiles = all_tissue_tiles
            self.all_image_file_names = all_image_file_names
            self.all_patient_barcodes = all_patient_barcodes
            self.all_image_ids = all_image_ids
            if self.target_kind == 'survival':
                self.all_censored = all_censored
                self.all_time_targets = all_time_targets
                self.all_binary_targets = all_binary_targets
                self.all_is_excluded = all_is_excluded
            else:
                self.all_targets = all_targets

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.slides = []
        self.grid_lists = []
        self.presaved_tiles = []

        if self.target_kind == 'survival':
            self.target_binary, self.target_time, self.censored = [], [], []

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                if self.target_kind == 'survival':
                    if all_censored[index] == 1:
                        censor_status = True
                    elif all_censored[index] == 0:
                        censor_status = False
                    else:
                        pass

                    self.censored.append(censor_status)
                    self.target_binary.append(all_binary_targets[index])
                    self.target_time.append(all_time_targets[index])
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(self.dir_dict[all_image_ids[index]])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.magnification.append(all_magnifications[index])
                self.presaved_tiles.append(all_image_ids[index] == 'ABCTB_TILES')
                if self.target_kind == 'survival':
                    self.target.append(-1)
                else:
                    self.target.append(all_targets[index])

                # Preload slides - improves speed during training.
                grid_file = []
                image_file = []
                try:
                    image_file = os.path.join(self.dir_dict[all_image_ids[index]], all_image_file_names[index])
                    if self.presaved_tiles[-1]:
                        tiles_dir = os.path.join(self.dir_dict[all_image_ids[index]], 'tiles',
                                                 '.'.join((os.path.basename(image_file)).split('.')[:-1]))
                        self.slides.append(tiles_dir)
                        self.grid_lists.append(0)
                    else:
                        # if self.train_type == 'Infer_All_Folds':
                        if self.train_type in ['Infer_All_Folds', 'Infer']:
                            self.slides.append(image_file)
                        else:
                            self.slides.append(openslide.open_slide(image_file))

                        basic_file_name = '.'.join(all_image_file_names[index].split('.')[:-1])

                        grid_file = os.path.join(self.dir_dict[all_image_ids[index]],
                                                 'Grids_' + str(self.desired_magnification),
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            self.grid_lists.append(grid_list)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Couldn\'t open slide {} or its Grid file {}'.format(image_file, grid_file))

        # Setting the transformation:
        self.transform = define_transformations(transform_type, self.train, self.tile_size, self.color_param)
        if np.sum(self.presaved_tiles):
            self.rand_crop = transforms.RandomCrop(self.tile_size)

        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()

        # Deleting attributes not needed.
        if train_type == 'Infer_All_Folds':
            attributes_to_delete = ['image_file_names', 'image_path_names', 'slides', 'presaved_tiles']
            for attribute in attributes_to_delete:
                delattr(self, attribute)

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        tiles, time_list, label, locs = _choose_data(grid_list=self.grid_lists[idx],
                                                     slide=self.slides[idx],
                                                     how_many=self.bag_size,
                                                     magnification=self.magnification[idx],
                                                     tile_size=self.tile_size,
                                                     print_timing=True,
                                                     desired_mag=self.desired_magnification,
                                                     random_shift=self.train)

        if self.target_kind == 'survival':
            if self.target_binary[idx] == 'Positive':
                label_binary = [1]
            elif self.target_binary[idx] == 'Negative':
                label_binary = [0]
            else:
                label_binary = [-1]

            label_binary = torch.LongTensor(label_binary)

            label_time = [self.target_time[idx]]
            label_time = torch.FloatTensor(label_time)
        else:
            label = [1] if self.target[idx] == 'Positive' else [0]
            label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = self.transform(tiles[i])

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.Compose([transforms.CenterCrop(self.tile_size), transforms.ToTensor()])
            for i in range(self.bag_size):
                images[i] = trans(tiles[i])
        else:
            images = 0

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        time_list = (time_list[0], time_list[1], aug_time, total_time)
        time_dict = {'Average time to extract a tile': time_list[1],
                     'Augmentation time': aug_time,
                     'Total time': total_time
                     }

        return {'Data': X,
                'Target': label,
                'Time List': time_list,
                'Time dict': time_dict,
                'File Names': self.image_file_names[idx],
                'Images': images,
                'Binary Target': label_binary,
                'Time Target': self.target_time[idx] if self.target_kind == 'survival' else torch.LongTensor([-1]),
                'Censored': bool(self.censored[idx]) if self.target_kind == 'survival' else torch.LongTensor([-1]),
                'Tile Locations': locs
                }


class WSI_REGdataset_Survival_CR(WSI_Master_Dataset_Survival_CR):
    def __init__(self,
                 DataSet: str = 'ABCTB',
                 tile_size: int = 256,
                 target_kind: str = 'survival',
                 test_fold: int = 1,
                 train: bool = True,
                 transform_type: str = 'none',
                 DX: bool = False,
                 get_images: bool = False,
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 10
                 ):
        super(WSI_REGdataset_Survival_CR, self).__init__(DataSet=DataSet,
                                                         tile_size=tile_size,
                                                         bag_size=1,
                                                         target_kind=target_kind,
                                                         test_fold=test_fold,
                                                         train=train,
                                                         transform_type=transform_type,
                                                         DX=DX,
                                                         get_images=get_images,
                                                         train_type='REG',
                                                         color_param=color_param,
                                                         n_tiles=n_tiles,
                                                         desired_slide_magnification=desired_slide_magnification
                                                         )

        logging.info(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
                .format(self.train_type,
                        'Train' if self.train else 'Test',
                        self.DataSet,
                        self.target_kind,
                        self.desired_magnification,
                        self.real_length,
                        self.tile_size,
                        self.bag_size,
                        'Without' if transform_type == 'none' else 'With',
                        self.test_fold,
                        'ON' if self.DX else 'OFF'))
        censored = np.array(self.censored)
        logging.info('{} Censored slides, {} Not Censored slides'.format(len(np.where(censored == True)[0]),
                                                                         len(np.where(censored == False)[0])))

    def __getitem__(self, idx):
        data_dict = super(WSI_REGdataset_Survival_CR, self).__getitem__(idx=idx)
        X = data_dict['Data']
        X = torch.reshape(X, (3, self.tile_size, self.tile_size))

        return {'Data': X,
                'Target': data_dict['Target'],
                'Censored': data_dict['Censored'],
                'Binary Target': data_dict['Binary Target'],
                'Time Target': data_dict['Time Target'],
                'Time dict': data_dict['Time dict'],
                'File Names': data_dict['File Names'],
                'Images': data_dict['Images'],
                'Tile Locations': data_dict['Tile Locations'][0]
                }


class WSI_Master_Dataset_Survival(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 test_time_augmentation: bool = False,
                 desired_slide_magnification: int = 10,
                 slide_repetitions: int = 1,
                 loan: bool = False,
                 er_eq_pr: bool = False,
                 slide_per_block: bool = False,
                 balanced_dataset: bool = False,
                 is_all_censored: bool = False,
                 is_all_not_censored: bool = False,
                 legit_slide_list: list = []):

        # Support multitarget training
        # Multi target training
        N_targets = len(target_kind.split('+'))
        if N_targets > 1:
            target_kind = target_kind.split('+')
            self.multi_target = True
        else:
            self.multi_target = False

        # Check if the target receptor is available for the requested train DataSet:
        assert_dataset_target(DataSet, target_kind)

        self._data_validity_check(is_all_censored, is_all_not_censored)

        print('Initializing {} DataSet....'.format('Train' if train else 'Test'))
        self.DataSet = DataSet
        self.desired_magnification = desired_slide_magnification
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.DX = DX
        self.get_images = get_images
        self.train_type = train_type
        self.color_param = color_param
        self.loan = loan

        if is_all_censored != is_all_not_censored:  # Checking that one of those flags is true
            self.censor_description = 'Censored' if is_all_censored else 'Not Censored'
        else:
            self.censor_description = 'As Is'

        # Get DataSets location:
        self.dir_dict = get_datasets_dir_dict(Dataset=self.DataSet)
        self._create_meta_data_DF()

        if self.DataSet == 'PORTO_HE' or self.DataSet == 'PORTO_PDL1':
            # for lung, take only origin: lung
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF.reset_index(inplace=True)

        if self._is_balanced_dataset_and_target_ER(balanced_dataset):
            self.meta_data_DF = balance_dataset(self.meta_data_DF)
            # take only selected patients
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['use_in_balanced_data_ER'] == 1]
            self.meta_data_DF.reset_index(inplace=True)

        if self._is_double_target_train__PR_or_ER():
            all_targets = self._set_all_targets_for_double_target_train()

        if self.target_kind == 'survival':
            all_censored = list(self.meta_data_DF['Censored'])
            all_time_targets = list(self.meta_data_DF['Follow-up Months Since Diagnosis'])
            all_binary_targets = list(self.meta_data_DF['survival status'])
            all_is_excluded = list(self.meta_data_DF['Exclude for time prediction?'])

        else:
            all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        all_patient_barcodes = list(self.meta_data_DF['patient barcode'])

        if slide_per_block:
            if DataSet == 'CARMEL':
                all_blocks = []
                for barcode in all_patient_barcodes:
                    if barcode is not np.nan:
                        all_blocks.append(barcode[:-2])
                    else:
                        all_blocks.append(barcode)

                _, unique_inds = np.unique(all_blocks, return_index=True)
                all_inds = np.arange(0, len(all_blocks))
                excess_block_slides = set(all_inds) - set(unique_inds)
                print('slide_per_block: removing ' + str(len(excess_block_slides)) + ' slides')
            else:
                IOError('slide_per_block only implemented for CARMEL dataset')
        elif (DataSet in ['LEUKEMIA', 'ALL']) and (
                target_kind in ['AML', 'ALL', 'is_B', 'is_HR', 'is_over_6', 'is_over_10', 'is_over_15', 'WBC_over_20',
                                'WBC_over_50', 'is_HR_B', 'is_tel_aml_B', 'is_tel_aml_non_hr_B']):
            # remove slides with diagnosis day != 0
            excess_block_slides = set(self.meta_data_DF.index[self.meta_data_DF['Day_0/15/33_fixed'] != 0])
        else:
            excess_block_slides = set()

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        if self.target_kind == 'survival':
            # In the case where the target is survival we need to exclude slides that are:
            # 1) Excluded
            # 2) without Censor data

            valid_slide_indices_exclude = np.where(np.array(all_is_excluded) != 'Exclude')[0]
            valid_slide_indices_censor = np.where(np.isnan(all_censored) == False)[0]
            # Combining all those conditions:
            valid_slide_indices = np.intersect1d(valid_slide_indices_exclude, valid_slide_indices_censor)

        else:
            valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == 0])

        if 'bad segmentation' in self.meta_data_DF.columns:
            slides_with_bad_seg = set(self.meta_data_DF.index[self.meta_data_DF['bad segmentation'] == 1])
        else:
            slides_with_bad_seg = set()

        # train only on samples with ER=PR
        if er_eq_pr and self.train:
            slides_with_er_not_eq_pr = set(
                self.meta_data_DF.index[self.meta_data_DF['ER status'] != self.meta_data_DF['PR status']])
        else:
            slides_with_er_not_eq_pr = set()

        # Define number of tiles to be used
        if train_type == 'REG':
            n_minimal_tiles = n_tiles
        else:
            n_minimal_tiles = self.bag_size

        slides_with_few_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] < n_minimal_tiles])
        # FIXME: find a way to use slides with less than the minimal amount of slides. and than delete the following if.
        if len(slides_with_few_tiles) > 0:
            print(
                '{} Slides were excluded from DataSet because they had less than {} available tiles or are non legitimate for training'
                    .format(len(slides_with_few_tiles), n_minimal_tiles))
        valid_slide_indices = np.array(
            list(set(
                valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles - slides_with_bad_seg - slides_with_er_not_eq_pr - excess_block_slides))

        # The train set should be a combination of all sets except the test set and validation set:
        if self.DataSet == 'CAT' or self.DataSet == 'ABCTB_TCGA':
            fold_column_name = 'test fold idx breast'
        else:
            fold_column_name = 'test fold idx'

        if self.train_type in ['REG', 'MIL']:
            if self.train:
                folds = list(self.meta_data_DF[fold_column_name].unique())
                folds.remove(self.test_fold)
                if 'test' in folds:
                    folds.remove('test')
                if 'val' in folds:
                    folds.remove('val')
            else:
                folds = [self.test_fold]
                folds.append('val')
        elif self.train_type == 'Infer':
            folds = infer_folds
        elif self.train_type == 'Infer_All_Folds':
            folds = list(self.meta_data_DF[fold_column_name].unique())
        else:
            raise ValueError('Variable train_type is not defined')

        self.folds = folds

        if type(folds) is int:
            folds = [folds]

        correct_folds = self.meta_data_DF[fold_column_name][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_ids = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
            self.desired_magnification)])

        if 'TCGA' not in self.dir_dict:
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Manipulated Objective Power'])

        if train_type in ['Infer', 'Infer_All_Folds']:
            temp_select = False
            if temp_select or len(legit_slide_list) > 0:  # hard coded selection of specific slides
                if temp_select:
                    slidenames = ['19-14722_1_1_a.mrxs', '19-14722_1_1_b.mrxs', '19-14722_1_1_e.mrxs',
                                  '19-5229_2_1_a.mrxs',
                                  '19-5229_2_1_b.mrxs', '19-5229_2_1_e.mrxs']
                else:
                    slidenames = legit_slide_list

                valid_slide_indices = []
                for slidename in slidenames:
                    valid_slide_index = self.meta_data_DF[self.meta_data_DF['file'] == slidename].index.to_list()
                    valid_slide_indices.append(valid_slide_index[0])

            self.valid_slide_indices = valid_slide_indices
            self.all_magnifications = all_magnifications
            self.all_is_DX_cut = all_is_DX_cut if self.DX else [True] * len(self.all_magnifications)
            self.all_tissue_tiles = all_tissue_tiles
            self.all_image_file_names = all_image_file_names
            self.all_patient_barcodes = all_patient_barcodes
            self.all_image_ids = all_image_ids
            if self.target_kind == 'survival':
                self.all_censored = all_censored
                self.all_time_targets = all_time_targets
                self.all_binary_targets = all_binary_targets
                self.all_is_excluded = all_is_excluded
            else:
                self.all_targets = all_targets

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.slides = []
        self.grid_lists = []
        self.presaved_tiles = []

        if self.target_kind == 'survival':
            self.target_binary, self.target_time, self.censored = [], [], []

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                if self.target_kind == 'survival':
                    if all_censored[index] == 1:
                        censor_status = True
                    elif all_censored[index] == 0:
                        censor_status = False
                    else:
                        pass
                    if (is_all_not_censored and censor_status) or (is_all_censored and not censor_status):
                        continue

                    self.censored.append(censor_status)
                    self.target_binary.append(all_binary_targets[index])
                    self.target_time.append(all_time_targets[index])
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(self.dir_dict[all_image_ids[index]])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.magnification.append(all_magnifications[index])
                self.presaved_tiles.append(all_image_ids[index] == 'ABCTB_TILES')
                if self.target_kind == 'survival':
                    self.target.append(-1)
                else:
                    self.target.append(all_targets[index])

                # Preload slides - improves speed during training.
                grid_file = []
                image_file = []
                try:
                    image_file = os.path.join(self.dir_dict[all_image_ids[index]], all_image_file_names[index])
                    if self.presaved_tiles[-1]:
                        tiles_dir = os.path.join(self.dir_dict[all_image_ids[index]], 'tiles',
                                                 '.'.join((os.path.basename(image_file)).split('.')[:-1]))
                        self.slides.append(tiles_dir)
                        self.grid_lists.append(0)
                    else:
                        # if self.train_type == 'Infer_All_Folds':
                        if self.train_type in ['Infer_All_Folds', 'Infer']:
                            self.slides.append(image_file)
                        else:
                            self.slides.append(openslide.open_slide(image_file))

                        basic_file_name = '.'.join(all_image_file_names[index].split('.')[:-1])

                        grid_file = os.path.join(self.dir_dict[all_image_ids[index]],
                                                 'Grids_' + str(self.desired_magnification),
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            self.grid_lists.append(grid_list)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Couldn\'t open slide {} or its Grid file {}'.format(image_file, grid_file))

        # Setting the transformation:
        self.transform = define_transformations(transform_type, self.train, self.tile_size, self.color_param)
        if np.sum(self.presaved_tiles):
            self.rand_crop = transforms.RandomCrop(self.tile_size)

        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train is False and test_time_augmentation:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)

        # Deleting attributes not needed.
        if train_type == 'Infer_All_Folds':
            attributes_to_delete = ['image_file_names', 'image_path_names', 'slides', 'presaved_tiles']
            for attribute in attributes_to_delete:
                delattr(self, attribute)

    def _create_meta_data_DF(self):
        print('Slide Data will be taken from these locations:')
        print(self.dir_dict)
        locations_list = []
        for _, key in enumerate(self.dir_dict):
            locations_list.append(self.dir_dict[key])

            slide_meta_data_file = os.path.join(self.dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(self.dir_dict[key], 'Grids_' + str(self.desired_magnification),
                                               'Grid_data.xlsx')

            slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
            grid_meta_data_DF = pd.read_excel(grid_meta_data_file)
            meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
                                         **grid_meta_data_DF.set_index('file').to_dict()})

            self.meta_data_DF = meta_data_DF if not hasattr(self, 'meta_data_DF') else self.meta_data_DF.append(
                meta_data_DF)
        self.meta_data_DF.reset_index(inplace=True)
        self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)

    def _data_validity_check(self, is_all_censored, is_all_not_censored):
        if is_all_censored and is_all_not_censored:
            raise Exception('\'is_all_censored\' and \'is_all_not_censored\' CANNOT be TRUE at the same time')

    def _is_double_target_train__PR_or_ER(self):
        return self.target_kind == 'OR'

    def _set_all_targets_for_double_target_train(self):
        PR_targets = list(self.meta_data_DF['PR status'])
        ER_targets = list(self.meta_data_DF['ER status'])
        all_targets = ['Missing Data'] * len(ER_targets)
        for ii, (PR_target, ER_target) in enumerate(zip(PR_targets, ER_targets)):
            if (PR_target == 'Positive' or ER_target == 'Positive'):
                all_targets[ii] = 'Positive'
            elif (PR_target == 'Negative' or ER_target == 'Negative'):  # avoid 'Missing Data'
                all_targets[ii] = 'Negative'
        return all_targets

    def _is_balanced_dataset_and_target_ER(self, is_balanced_dataset):
        return is_balanced_dataset and self.target_kind == 'ER'

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        if self.presaved_tiles[idx]:  # load presaved patches
            time_tile_extraction = time.time()
            idxs = sample(range(self.tissue_tiles[idx]), self.bag_size)
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * self.bag_size
            for ii, tile_ind in enumerate(idxs):
                tile_path = os.path.join(self.slides[idx], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
                time_tile_extraction = (time.time() - time_tile_extraction) / len(idxs)
                time_list = [0, time_tile_extraction]
        else:
            slide = self.slides[idx]

            tiles, time_list, label, locs = _choose_data(grid_list=self.grid_lists[idx],
                                                         slide=slide,
                                                         how_many=self.bag_size,
                                                         magnification=self.magnification[idx],
                                                         tile_size=self.tile_size,
                                                         print_timing=True,
                                                         desired_mag=self.desired_magnification,
                                                         loan=self.loan,
                                                         random_shift=self.train)

        if not self.loan:
            if self.target_kind == 'survival':
                if self.target_binary[idx] == 'Positive':
                    label_binary = [1]
                elif self.target_binary[idx] == 'Negative':
                    label_binary = [0]
                else:
                    label_binary = [-1]

                label_binary = torch.LongTensor(label_binary)

                label_time = [self.target_time[idx]]
                label_time = torch.FloatTensor(label_time)
            else:
                label = [1] if self.target[idx] == 'Positive' else [0]
                label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = self.transform(tiles[i])

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.Compose([transforms.CenterCrop(self.tile_size), transforms.ToTensor()])
            for i in range(self.bag_size):
                images[i] = trans(tiles[i])
        else:
            images = 0

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        time_list = (time_list[0], time_list[1], aug_time, total_time)
        time_dict = {'Average time to extract a tile': time_list[1],
                     'Augmentation time': aug_time,
                     'Total time': total_time
                     }

        return {'Data': X,
                'Target': label,
                'Time List': time_list,
                'Time dict': time_dict,
                'File Names': self.image_file_names[idx],
                'Images': images,
                'Binary Target': label_binary,
                'Time Target': self.target_time[idx] if self.target_kind == 'survival' else torch.LongTensor([-1]),
                'Censored': bool(self.censored[idx]) if self.target_kind == 'survival' else torch.LongTensor([-1]),
                'Tile Locations': locs
                }


class Infer_Dataset_Survival(WSI_Master_Dataset_Survival):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 num_tiles: int = 500,
                 dx: bool = False,
                 desired_slide_magnification: int = 10,
                 resume_slide: int = 0,
                 patch_dir: str = ''
                 ):
        if patch_dir != '':
            xfile = glob(os.path.join(patch_dir, '*_x.csv'))
            if len(xfile) == 0:
                raise IOError('patch location files not found in dir!')
            elif len(xfile) > 1:
                raise IOError('more than one patch location file in dir!')
            slides_with_locations = pd.read_csv(xfile[0])['slide_name']

        else:
            slides_with_locations = []

        super(Infer_Dataset_Survival, self).__init__(DataSet=DataSet,
                                                     tile_size=tile_size,
                                                     bag_size=None,
                                                     target_kind=target_kind,
                                                     test_fold=1,
                                                     infer_folds=folds,
                                                     train=True,
                                                     transform_type='none',
                                                     DX=dx,
                                                     train_type='Infer',
                                                     desired_slide_magnification=desired_slide_magnification,
                                                     legit_slide_list=slides_with_locations)

        self.print_time = False
        self.tiles_per_iter = tiles_per_iter
        self.patch_dir = patch_dir
        self.folds = folds
        self.magnification = []
        self.num_tiles = []
        self.slide_grids = []
        self.grid_lists = []
        self.patient_barcode = []
        self.slide_dataset = []

        ind = 0
        slide_with_not_enough_tiles = 0

        self.valid_slide_indices = self.valid_slide_indices[resume_slide:]
        self.tissue_tiles = self.tissue_tiles[resume_slide:]
        self.image_file_names = self.image_file_names[resume_slide:]
        self.image_path_names = self.image_path_names[resume_slide:]
        self.slides = self.slides[resume_slide:]
        self.presaved_tiles = self.presaved_tiles[resume_slide:]
        self.target = self.target[resume_slide:]

        if self.patch_dir != '':
            # load patches position from excel file
            xfile = glob(os.path.join(self.patch_dir, '*_x.csv'))
            yfile = glob(os.path.join(self.patch_dir, '*_y.csv'))
            if len(xfile) == 0 or len(yfile) == 0:
                raise IOError('patch location files not found in dir!')
            elif len(xfile) > 1 or len(yfile) > 1:
                raise IOError('more than one patch location file in dir!')
            self.x_pd = pd.read_csv(xfile[0])
            self.y_pd = pd.read_csv(yfile[0])

        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= self.all_tissue_tiles[slide_num] and self.all_tissue_tiles[slide_num] > 0:
                    self.num_tiles.append(num_tiles)
                else:
                    self.num_tiles.append(int(self.all_tissue_tiles[slide_num]))
                    slide_with_not_enough_tiles += 1

                self.magnification.extend([self.all_magnifications[slide_num]])
                self.patient_barcode.append(self.all_patient_barcodes[slide_num])
                self.slide_dataset.append(self.all_image_ids[slide_num])
                if self.patch_dir == '':
                    which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])
                else:
                    which_patches = [ii for ii in range(self.num_tiles[-1])]

                patch_ind_chunks = chunks(which_patches, self.tiles_per_iter)
                self.slide_grids.extend(patch_ind_chunks)

                if self.presaved_tiles[ind]:
                    self.grid_lists.append(0)
                else:
                    if self.patch_dir == '':
                        basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                        grid_file = os.path.join(self.image_path_names[ind], 'Grids_' + str(self.desired_magnification),
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            self.grid_lists.append(grid_list)

                ind += 1

        logging.info('There are {} slides with less than {} tiles'.format(slide_with_not_enough_tiles, num_tiles))

        # The following properties will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = -1
        self.current_file = None

        print('Initiation of WSI INFERENCE for {} DataSet and {} of folds {} is Complete'
              .format(self.DataSet,
                      self.target_kind,
                      str(self.folds)))

        print('{} Slides, with X{} magnification. {} tiles per iteration, {} iterations to complete full inference'
              .format(len(self.image_file_names),
                      self.desired_magnification,
                      self.tiles_per_iter,
                      self.__len__()))

    def __len__(self):
        return int(np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.slide_num += 1
            self.tiles_to_go = self.num_tiles[self.slide_num]
            self.slide_name = self.image_file_names[self.slide_num]
            self.current_slide = openslide.OpenSlide(self.slides[self.slide_num])
            self.initial_num_patches = self.num_tiles[self.slide_num]

            if not self.presaved_tiles[self.slide_num]:
                self.best_slide_level, self.adjusted_tile_size, self.level_0_tile_size = \
                    get_optimal_slide_level(self.current_slide, self.magnification[self.slide_num],
                                            self.desired_magnification, self.tile_size)

        label = get_label(self.target[self.slide_num], self.multi_target)
        label = torch.LongTensor(label)

        if self.target_kind == 'survival':
            target_binary_current = get_label(self.target_binary[self.slide_num])
            target_binary_current = torch.LongTensor(target_binary_current)
            target_time_current = self.target_time[self.slide_num]
            censor_status_current = self.censored[self.slide_num]

        else:
            target_binary_current, target_time_current, censor_status_current = -1, -1, -1

        if self.presaved_tiles[self.slide_num]:
            idxs = self.slide_grids[idx]
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * len(idxs)
            for ii, tile_ind in enumerate(idxs):
                # tile_path = os.path.join(self.tiles_dir[idx], 'tile_' + str(tile_ind) + '.data')
                tile_path = os.path.join(self.slides[self.slide_num], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
        else:
            if self.patch_dir == '':
                locs = [self.grid_lists[self.slide_num][loc] for loc in self.slide_grids[idx]]

            else:
                x_loc = [int(self.x_pd.loc[self.x_pd['slide_name'] == self.slide_name][str(loc)].item()) for loc in
                         self.slide_grids[idx]]
                y_loc = [int(self.y_pd.loc[self.y_pd['slide_name'] == self.slide_name][str(loc)].item()) for loc in
                         self.slide_grids[idx]]

                locs = [ii for ii in zip(x_loc, y_loc)]

            tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                             # locations=self.slide_grids[idx],
                                             locations=locs,
                                             tile_size_level_0=self.level_0_tile_size,
                                             adjusted_tile_sz=self.adjusted_tile_size,
                                             output_tile_sz=self.tile_size,
                                             best_slide_level=self.best_slide_level,
                                             random_shift=False)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
        else:
            self.tiles_to_go -= self.tiles_per_iter

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = self.transform(tiles[i])

        aug_time = time.time() - start_aug
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]
        if self.tiles_to_go is None:
            last_batch = True
        else:
            last_batch = False

        return {'Data': X,
                'Label': label,  # It is acutally the target and not the label
                'Time List': time_list,
                'Is Last Batch': last_batch,
                'Initial Num Tiles': self.initial_num_patches,
                'Slide Filename': self.slide_name,
                'Patient barcode': self.patient_barcode[self.slide_num],
                'Slide DataSet': self.slide_dataset[self.slide_num],
                'Patch Loc': locs,
                'Censored': censor_status_current,
                'Binary Target': target_binary_current,
                'Time Target': target_time_current
                }


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.len_0, self.len_1 = len(self.datasets[0]), len(self.datasets[1])
        self.total_len = self.len_0 + self.len_1

    def __getitem__(self, i):
        # return tuple(d[i] for d in self.datasets)
        # return tuple(d[i % len(d)] for d in self.datasets)
        if i < self.len_0:
            return self.datasets[0][i]
        else:
            return self.datasets[1][i]

    def __len__(self):
        # return min(len(d) for d in self.datasets)
        # return max(len(d) for d in self.datasets)
        return self.total_len