import numpy as np
from PIL import Image
from matplotlib import image as plt_image
import os
import pandas as pd
import glob
from random import sample, seed
import torch
import time
from typing import List, Tuple
from xlrd.biffh import XLRDError
from zipfile import BadZipFile
import matplotlib.pyplot as plt
from math import isclose
from argparse import Namespace as argsNamespace
from shutil import copyfile
from datetime import date
import inspect
import torch.nn.functional as F
import multiprocessing
from tqdm import tqdm
import sys
from PIL import ImageFile
from fractions import Fraction
import openslide
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def chunks(list: List, length: int):
    new_list = [list[i * length:(i + 1) * length] for i in range((len(list) + length - 1) // length)]
    return new_list


def get_optimal_slide_level(slide, magnification, desired_mag, tile_size):
    # downsample needed for each dimension (reflected by level_downsamples property)
    desired_downsample = magnification / desired_mag

    if desired_downsample < 1:  # upsample
        best_slide_level = 0
        level_0_tile_size = int(desired_downsample * tile_size)
        adjusted_tile_size = level_0_tile_size
    else:
        level, best_next_level = -1, -1
        for index, downsample in enumerate(slide.level_downsamples):
            if isclose(desired_downsample, downsample, rel_tol=1e-3):
                level = index
                level_downsample = 1
                break

            elif downsample < desired_downsample:
                best_next_level = index
                level_downsample = desired_downsample / slide.level_downsamples[best_next_level]

        adjusted_tile_size = int(tile_size * level_downsample)
        best_slide_level = level if level > best_next_level else best_next_level
        level_0_tile_size = int(desired_downsample * tile_size)

    return best_slide_level, adjusted_tile_size, level_0_tile_size


def _choose_data(grid_list: list,
                 slide: openslide.OpenSlide,
                 how_many: int,
                 magnification: int,
                 tile_size: int = 256,
                 print_timing: bool = False,
                 desired_mag: int = 20,
                 loan: bool = False,
                 random_shift: bool = True):
    """
    This function choose and returns data to be held by DataSet.
    The function is in the PreLoad Version. It works with slides already loaded to memory.

    :param grid_list: A list of all grids for this specific slide
    :param slide: An OpenSlide object of the slide.
    :param how_many: how_many tiles to return from the slide.
    :param magnification: The magnification of level 0 of the slide
    :param tile_size: Desired tile size from the slide at the desired magnification
    :param print_timing: Do or don't collect timing for this procedure
    :param desired_mag: Desired Magnification of the tiles/slide.
    :return:
    """

    best_slide_level, adjusted_tile_size, level_0_tile_size = get_optimal_slide_level(slide, magnification, desired_mag,
                                                                                      tile_size)

    # Choose locations from the grid list:
    loc_num = len(grid_list)
    try:
        idxs = sample(range(loc_num), how_many)
    except ValueError:
        raise ValueError('Requested more tiles than available by the grid list')

    locs = [grid_list[idx] for idx in idxs]
    image_tiles, time_list, labels = _get_tiles(slide=slide,
                                                locations=locs,
                                                tile_size_level_0=level_0_tile_size,
                                                adjusted_tile_sz=adjusted_tile_size,
                                                output_tile_sz=tile_size,
                                                best_slide_level=best_slide_level,
                                                print_timing=print_timing,
                                                random_shift=random_shift,
                                                loan=loan)

    return image_tiles, time_list, labels, locs


def _get_tiles(slide: openslide.OpenSlide,
               locations: List[Tuple],
               tile_size_level_0: int,
               adjusted_tile_sz: int,
               output_tile_sz: int,
               best_slide_level: int,
               print_timing: bool = False,
               random_shift: bool = False,
               oversized_HC_tiles: bool = False,
               loan: bool = False):
    """
    This function extract tiles from the slide.
    :param slide: OpenSlide object containing a slide
    :param locations: locations of te tiles to be extracted
    :param tile_size_level_0: tile size adjusted for level 0
    :param adjusted_tile_sz: tile size adjusted for best_level magnification
    :param output_tile_sz: output tile size needed
    :param best_slide_level: best slide level to get tiles from
    :param print_timing: collect time profiling data ?
    :return:
    """

    # preallocate list of images
    empty_image = Image.fromarray(np.uint8(np.zeros((output_tile_sz, output_tile_sz, 3))))
    tiles_PIL = [empty_image] * len(locations)

    start_gettiles = time.time()

    if oversized_HC_tiles:
        adjusted_tile_sz *= 2
        output_tile_sz *= 2
        tile_shifting = (tile_size_level_0 // 2, tile_size_level_0 // 2)

    # get localized labels
    labels = np.zeros(len(locations)) - 1
    if loan:
        slide_name = os.path.splitext(os.path.basename(slide._filename))[0]
        annotation_file = os.path.join(os.path.dirname(slide._filename), 'local_labels', slide_name + '-labels.png')
        annotation = (plt_image.imread(annotation_file) * 255).astype('uint8')
        ds = 8  # defined in the QuPath groovy script

    for idx, loc in enumerate(locations):
        if random_shift:
            tile_shifting = sample(range(-tile_size_level_0 // 2, tile_size_level_0 // 2), 2)

        if random_shift or oversized_HC_tiles:
            new_loc_init = {'Top': loc[0] - tile_shifting[0],
                            'Left': loc[1] - tile_shifting[1]}
            new_loc_end = {'Bottom': new_loc_init['Top'] + tile_size_level_0,
                           'Right': new_loc_init['Left'] + tile_size_level_0}
            if new_loc_init['Top'] < 0:
                new_loc_init['Top'] += abs(new_loc_init['Top'])
            if new_loc_init['Left'] < 0:
                new_loc_init['Left'] += abs(new_loc_init['Left'])
            if new_loc_end['Bottom'] > slide.dimensions[1]:
                delta_Height = new_loc_end['Bottom'] - slide.dimensions[1]
                new_loc_init['Top'] -= delta_Height
            if new_loc_end['Right'] > slide.dimensions[0]:
                delta_Width = new_loc_end['Right'] - slide.dimensions[0]
                new_loc_init['Left'] -= delta_Width
        else:
            new_loc_init = {'Top': loc[0],
                            'Left': loc[1]}

        try:
            image = slide.read_region((new_loc_init['Left'], new_loc_init['Top']), best_slide_level,
                                      (adjusted_tile_sz, adjusted_tile_sz)).convert('RGB')
        except:
            logging.info('failed to read slide {} in location {},{}'.format(slide._filename, loc[1], loc[0]))
            logging.info('taking blank patch instead')
            image = Image.fromarray(np.zeros([adjusted_tile_sz, adjusted_tile_sz, 3], dtype=np.uint8))

        # get localized labels
        if loan:
            d = adjusted_tile_sz // ds
            x = new_loc_init['Left'] // ds
            y = new_loc_init['Top'] // ds
            x0 = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_X]) // ds
            y0 = int(slide.properties[openslide.PROPERTY_NAME_BOUNDS_Y]) // ds

            annotation_tile = annotation[y - y0:y - y0 + d, x - x0:x - x0 + d, :]
            red_zone = np.sum(annotation_tile[:, :, 0] == 250) / (annotation_tile.size // 3)
            if red_zone > 0.1:
                labels[idx] = 1
            else:
                labels[idx] = 0

        if adjusted_tile_sz != output_tile_sz:
            image = image.resize((output_tile_sz, output_tile_sz))

        tiles_PIL[idx] = image

    end_gettiles = time.time()

    if print_timing:
        time_list = [0, (end_gettiles - start_gettiles) / len(locations)]
    else:
        time_list = [0]

    return tiles_PIL, time_list, labels


def device_gpu_cpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('Using CUDA')
    else:
        device = torch.device('cpu')
        logging.info('Using cpu')
    return device


def get_cpu():
    platform = sys.platform
    if platform == 'linux':
        cpu = len(os.sched_getaffinity(0))
    elif platform == 'darwin':
        cpu = 2
        platform = 'MacOs'
    else:  # windows
        cpu = multiprocessing.cpu_count()
        platform = 'Windows'

    logging.info('Running on {} with {} CPUs'.format(platform, cpu))
    return cpu


def run_data(experiment: str = None,
             test_fold: int = 1,
             transform_type: str = 'none',
             tile_size: int = 256,
             tiles_per_bag: int = 50,
             num_bags: int = 1,
             DX: bool = False,
             DataSet_name: list = ['TCGA'],
             DataSet_test_name: list = [None],
             DataSet_size: tuple = None,
             DataSet_Slide_magnification: int = None,
             epoch: int = None,
             model: str = None,
             transformation_string: str = None,
             Receptor: str = None,
             MultiSlide: bool = False,
             test_mean_auc: float = None,
             is_per_patient: bool = False,
             is_last_layer_freeze: bool = False,
             is_repeating_data: bool = False,
             data_limit: int = None,
             free_bias: bool = False,
             carmel_only: bool = False,
             CAT_only: bool = False,
             Remark: str = '',
             Class_Relation: float = None,
             learning_rate: float = -1,
             weight_decay: float = -1,
             censored_ratio: float = -1,
             combined_loss_weights: list = [],
             receptor_tumor_mode: int = -1,
             is_domain_adaptation: bool = False):
    """
    This function writes the run data to file
    :param experiment:
    :param from_epoch:
    :param MultiSlide: Describes if tiles from different slides with same class are mixed in the same bag
    :return:
    """

    if experiment is not None:
        if sys.platform == 'linux':
            if 0 < int(experiment) < 10000:
                # One of Ran's experiments
                user_name = 'Ran'
                run_file_name = r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/run_data.xlsx'
                location_prefix = '/home/rschley/code/WSI_MIL/WSI_MIL/'
            elif 10000 < int(experiment) < 20000:
                # One of Omer's experiments
                user_name = 'Omer'
                run_file_name = r'/home/womer/project/runs/run_data.xlsx'
                location_prefix = '/home/womer/project/'
            elif 20000 < int(experiment) < 30000:
                # One of Gil's experiments
                user_name = 'Gil'
                run_file_name = r'/mnt/gipnetapp_public/sgils/ran/runs/run_data.xlsx'
                location_prefix = '/mnt/gipnetapp_public/sgils/ran/'
            elif 30000 < int(experiment) < 40000:
                # One of Tal's experiments
                user_name = 'Tal'
                run_file_name = r'/home/talneoran/workspace/wsi/runs/run_data.xlsx'
                location_prefix = '/home/talneoran/workspace/wsi'
            elif 40000 < int(experiment) < 50000:
                # One of Hen's experiments
                user_name = 'dahen'
                run_file_name = r'/home/dahen/WSI_ran_legacy/WSI/runs/run_data.xlsx'
                location_prefix = '/home/dahen/WSI_ran_legacy/WSI'
            elif 50000 < int(experiment) < 60000:
                # One of Shachar's experiments
                user_name = 'Shachar'
                run_file_name = r'/home/shacharcohen/workspace/WSI/runs/run_data.xlsx'
                location_prefix = '/home/shacharcohen/workspace/WSI'
        else:
            user_name = None
            run_file_name = 'runs/run_data.xlsx'

    else:
        run_file_name = 'runs/run_data.xlsx'

    if sys.platform == 'win32':  # Ran's laptop
        run_file_name = r'C:\WSI_MIL_runs\run_data.xlsx'

    if os.path.isfile(run_file_name):
        read_success = False
        read_attempts = 0
        while (not read_success) and (read_attempts < 10):
            try:
                run_DF = pd.read_excel(run_file_name)
                read_success = True
            except (XLRDError, BadZipFile):
                print('Couldn\'t open file {}, check if file is corrupt'.format(run_file_name))
                return
            except ValueError:
                print('run_data file is being used, retrying in 5 seconds')
                read_attempts += 1
                time.sleep(5)
        if not read_success:
            print('Couldn\'t open file {} after 10 attempts'.format(run_file_name))
            return

        try:
            run_DF.drop(labels='Unnamed: 0', axis='columns', inplace=True)
        except KeyError:
            pass

        run_DF_exp = run_DF.set_index('Experiment', inplace=False)
    else:
        run_DF = pd.DataFrame()

    # If a new experiment is conducted:
    if experiment is None:
        if os.path.isfile(run_file_name):
            experiment = run_DF_exp.index.values.max() + 1
        else:
            experiment = 1

        location = os.path.join(os.path.abspath(os.getcwd()), 'runs',
                                'Exp_' + str(experiment) + '-' + Receptor + '-TestFold_' + str(test_fold))
        if type(DataSet_name) is not list:
            DataSet_name = [DataSet_name]

        if type(DataSet_test_name) is not list:
            DataSet_test_name = [DataSet_test_name]

        run_dict = {'Experiment': experiment,
                    'Start Date': str(date.today()),
                    'Test Fold': test_fold,
                    'Transformations': transform_type,
                    'Tile Size': tile_size,
                    'Tiles Per Bag': tiles_per_bag,
                    'MultiSlide Per Bag': MultiSlide,
                    'No. of Bags': num_bags,
                    'Location': location,
                    'DX': DX,
                    'DataSet': ' / '.join(DataSet_name),
                    'Test Set (DataSet)': ' / '.join(DataSet_test_name) if DataSet_test_name[0] != None else None,
                    'Receptor': Receptor,
                    'Model': 'None',
                    'Last Epoch': 0,
                    'Transformation String': 'None',
                    'Desired Slide Magnification': DataSet_Slide_magnification,
                    'Per Patient Training': is_per_patient,
                    'Last Layer Freeze': is_last_layer_freeze,
                    'Repeating Data': is_repeating_data,
                    'Data Limit': data_limit,
                    'Free Bias': free_bias,
                    'Carmel Only': carmel_only,
                    'Using Feature from CAT model alone': CAT_only,
                    'Remark': Remark,
                    'Class Relation': Class_Relation,
                    'Learning Rate': learning_rate,
                    'Weight Decay': weight_decay,
                    'Censor Ratio': censored_ratio,
                    'Combined Loss Weights': [Fraction(combined_loss_weights[item]).limit_denominator() for item in
                                              range(len(combined_loss_weights))],
                    'Receptor + is_Tumor Train Mode': receptor_tumor_mode,
                    'Trained with Domain Adaptation': is_domain_adaptation
                    }
        run_DF = run_DF.append([run_dict], ignore_index=True)
        if not os.path.isdir('runs'):
            os.mkdir('runs')

        if not os.path.isdir(location):
            os.mkdir(location)

        run_DF.to_excel(run_file_name)
        print('Created a new Experiment (number {}). It will be saved at location: {}'.format(experiment, location))

        # backup for run_data
        backup_dir = os.path.join(os.path.abspath(os.getcwd()), 'runs', 'run_data_backup')
        print(backup_dir)
        if not os.path.isdir(backup_dir):
            os.mkdir(backup_dir)
            print('backup dir created')
        try:
            run_DF.to_excel(os.path.join(backup_dir, 'run_data_exp' + str(experiment) + '.xlsx'))
        except:
            raise IOError('failed to back up run_data, please check there is enough storage')

        return {'Location': location,
                'Experiment': experiment
                }

    elif experiment is not None and epoch is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Last Epoch'] = epoch
        run_DF.to_excel(run_file_name)

    elif experiment is not None and model is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Model'] = model
        run_DF.to_excel(run_file_name)

    elif experiment is not None and transformation_string is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Transformation String'] = transformation_string
        run_DF.to_excel(run_file_name)

    elif experiment is not None and DataSet_size is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Train DataSet Size'] = DataSet_size[0]
        run_DF.at[index, 'Test DataSet Size'] = DataSet_size[1]
        run_DF.to_excel(run_file_name)

    elif experiment is not None and DataSet_Slide_magnification is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Desired Slide Magnification'] = DataSet_Slide_magnification
        run_DF.to_excel(run_file_name)

    elif experiment is not None and test_mean_auc is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'TestSet Mean AUC'] = test_mean_auc
        run_DF.to_excel(run_file_name)

    # In case we want to continue from a previous training session
    else:
        location = run_DF_exp.loc[[experiment], ['Location']].values[0][0]
        test_fold = int(run_DF_exp.loc[[experiment], ['Test Fold']].values[0][0])
        transformations = run_DF_exp.loc[[experiment], ['Transformations']].values[0][0]
        tile_size = int(run_DF_exp.loc[[experiment], ['Tile Size']].values[0][0])
        tiles_per_bag = int(run_DF_exp.loc[[experiment], ['Tiles Per Bag']].values[0][0])
        num_bags = int(run_DF_exp.loc[[experiment], ['No. of Bags']].values[0][0])
        DX = bool(run_DF_exp.loc[[experiment], ['DX']].values[0][0])
        DataSet_name = str(run_DF_exp.loc[[experiment], ['DataSet']].values[0][0])
        Receptor = str(run_DF_exp.loc[[experiment], ['Receptor']].values[0][0])
        MultiSlide = str(run_DF_exp.loc[[experiment], ['MultiSlide Per Bag']].values[0][0])
        model_name = str(run_DF_exp.loc[[experiment], ['Model']].values[0][0])
        Desired_Slide_magnification = int(run_DF_exp.loc[[experiment], ['Desired Slide Magnification']].values[0][0])
        try:
            receptor_tumor_mode = run_DF_exp.loc[[experiment], ['Receptor + is_Tumor Train Mode']].values[0][0]
            receptor_tumor_mode = convert_value_to_integer(receptor_tumor_mode)
            free_bias = bool(run_DF_exp.loc[[experiment], ['Free Bias']].values[0][0])
            CAT_only = bool(run_DF_exp.loc[[experiment], ['Using Feature from CAT model alone']].values[0][0])
            Class_Relation = float(run_DF_exp.loc[[experiment], ['Class Relation']].values[0][0])
        except:
            receptor_tumor_mode = np.nan
            free_bias = np.nan
            CAT_only = np.nan
            Class_Relation = np.nan

        if sys.platform == 'linux':
            if location.split('/')[0] == 'runs':
                location = location_prefix + location

        return {'Location': location,
                'Test Fold': test_fold,
                'Transformations': transformations,
                'Tile Size': tile_size,
                'Tiles Per Bag': tiles_per_bag,
                'Num Bags': num_bags,
                'DX': DX,
                'Dataset Name': DataSet_name,
                'Receptor': Receptor,
                'MultiSlide': MultiSlide,
                'Model Name': model_name,
                'Desired Slide Magnification': Desired_Slide_magnification,
                'Free Bias': free_bias,
                'CAT Only': CAT_only,
                'Class Relation': Class_Relation,
                'User': user_name,
                'Receptor + is_Tumor Train Mode': receptor_tumor_mode
                }


def convert_value_to_integer(value):
    return int(value) if not np.isnan(value) else -1


def assert_dataset_target(DataSet, target_kind):
    # Support multi targets
    if type(target_kind) != list:
        target_kind = [target_kind]
    target_kind = set(target_kind)

    if DataSet == 'TMA_HE_02_008' and not target_kind <= {'ER', 'temp', 'binary_dist', 'binary_live', 'binary_any'}:
        raise ValueError('For TMA_HE_02_008 DataSet, target should be one of: ER, binary_dist, binary_live, binary_any')
    if DataSet == 'TMA_HE_01_011' and not target_kind <= {'binary_live', 'ER'}:
        raise ValueError('For TMA_HE_01_011 DataSet, target should be one of: binary_live, ER')
    elif DataSet == 'PORTO_HE' and not target_kind <= {'PDL1', 'EGFR', 'is_full_cancer'}:
        raise ValueError('For PORTO_HE DataSet, target should be one of: PDL1, EGFR')
    elif DataSet == 'PORTO_PDL1' and not target_kind <= {'PDL1'}:
        raise ValueError('For PORTO_PDL1 DataSet, target should be PDL1')
    elif (DataSet in ['TCGA', 'CAT', 'ABCTB_TCGA']) and not target_kind <= {'ER', 'PR', 'Her2', 'OR', 'is_cancer',
                                                                            'Ki67'}:
        raise ValueError('target should be one of: ER, PR, Her2, OR, is_cancer, Ki67')
    elif (DataSet in ['IC', 'HIC', 'HEROHE', 'HAEMEK']) and not target_kind <= {'ER', 'PR', 'Her2', 'OR', 'Ki67'}:
        raise ValueError('target should be one of: ER, PR, Her2, OR')
    elif (DataSet == 'CARMEL') and not target_kind <= {'ER', 'PR', 'Her2', 'OR', 'Ki67', 'ER100'}:
        raise ValueError('target should be one of: ER, PR, Her2, OR')
    elif (DataSet == 'RedSquares') and not target_kind <= {'RedSquares'}:
        raise ValueError('target should be: RedSquares')
    elif DataSet == 'SHEBA' and not target_kind <= {'Onco', 'onco_score_11', 'onco_score_18', 'onco_score_26',
                                                    'onco_score_31', 'onco_score_all'}:
        raise ValueError('Invalid target for SHEBA DataSet')
    elif DataSet == 'TCGA_LUNG' and not target_kind <= {'is_cancer', 'is_LUAD', 'is_full_cancer'}:
        raise ValueError('for TCGA_LUNG DataSet, target should be is_cancer or is_LUAD')
    elif (DataSet in ['LEUKEMIA', 'ALL', 'AML']) and not target_kind <= {'ALL', 'is_B', 'is_HR', 'is_over_6',
                                                                         'is_over_10', 'is_over_15', 'WBC_over_20',
                                                                         'WBC_over_50', 'is_HR_B', 'is_tel_aml_B',
                                                                         'is_tel_aml_non_hr_B', 'MRD_day0', 'MRD_day15',
                                                                         'MRD_day33', 'MRD_all_days', 'AML',
                                                                         'provisional risk', 'provisional risk 10'}:
        raise ValueError('Invalid target for DataSet')
    elif (DataSet in ['ABCTB', 'ABCTB_TIF']) and not target_kind <= {'ER', 'PR', 'Her2', 'survival', 'Survival_Time',
                                                                     'Survival_Binary'}:
        raise ValueError('Invalid target for DataSet')
    elif (DataSet == 'CARMEL+BENIGN') and not target_kind <= {'is_cancer'}:
        raise ValueError('target should be is_cancer')


def save_code_files(args: argsNamespace, train_DataSet):
    """
    This function saves the code files and argparse data to a Code directory within the run path.
    :param args: argsparse Namespace of the run.
    :return:
    """
    code_files_path = os.path.join(args.output_dir, 'Code')
    # Get the filename that called this function
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    full_filename = module.__file__
    args.run_file = full_filename.split('/')[-1]

    args_dict = vars(args)

    # Add Grid Data:
    data_dict = args_dict
    if train_DataSet is not None:
        if type(train_DataSet) == dict:
            train_DataSet = train_DataSet['Censored']

        if train_DataSet.train_type != 'Features':
            for _, key in enumerate(train_DataSet.dir_dict):
                grid_meta_data_file = os.path.join(train_DataSet.dir_dict[key],
                                                   'Grids_' + str(train_DataSet.desired_magnification),
                                                   'production_meta_data.xlsx')
                if os.path.isfile(grid_meta_data_file):
                    grid_data_DF = pd.read_excel(grid_meta_data_file)
                    grid_dict = grid_data_DF.to_dict('split')
                    grid_dict['dataset'] = key
                    grid_dict.pop('index')
                    grid_dict.pop('columns')
                    data_dict[key + '_grid'] = grid_dict

    data_DF = pd.DataFrame([data_dict]).transpose()

    if not os.path.isdir(code_files_path):
        os.mkdir(code_files_path)
    data_DF.to_excel(os.path.join(code_files_path, 'run_arguments.xlsx'))
    py_files = glob.glob('*.py')  # Get all .py files in the code path
    for _, file in enumerate(py_files):
        copyfile(file, os.path.join(code_files_path, os.path.basename(file)))


def map_original_grid_list_to_equiv_grid_list(adjusted_tile_size, grid_list):
    """
    This function is used in datasets.Full_Slide_Inference_Dataset.
    It's use is to find the corresponding locations in the equivalent grid list of the tiles in the original_grid_list
    """
    equivalent_grid = []
    for location in grid_list:
        equivalent_location = (location[0] // adjusted_tile_size, location[1] // adjusted_tile_size)
        equivalent_grid.append(equivalent_location)

    return equivalent_grid


def balance_dataset(meta_data_DF, censor_balance: bool = False, test_fold=1):
    seed(2021)

    if censor_balance:
        # We need to balance the train set
        train_folds = set(meta_data_DF['test fold idx'])
        if 'test' in train_folds:
            train_folds.remove('test')
        if 'val' in train_folds:
            train_folds.remove('val')

        train_folds.remove(test_fold)

        # Now we'll remove all slides that do not belong to the train set:
        meta_data_DF = meta_data_DF[meta_data_DF['test fold idx'].isin(train_folds)]

        meta_data_DF['use_in_balanced_dataset'] = 0
        meta_data_DF.loc[meta_data_DF['Censored'] == 0, 'use_in_balanced_dataset'] = 1  # take all not censored

        not_censored_slides = meta_data_DF[meta_data_DF['Censored'] == 0]['file'].to_list()
        censored_slides = meta_data_DF[meta_data_DF['Censored'] == 1]['file'].to_list()

        # pick which censored slides to take:
        censored_patients_to_take = sample(censored_slides, k=len(not_censored_slides))
        logging.info('Not censored slides: {}'.format(len(not_censored_slides)))
        logging.info('Censored slides: {}'.format(len(censored_slides)))

        for patient_to_take in censored_patients_to_take:
            if len(meta_data_DF.loc[meta_data_DF['file'] == patient_to_take]) > 1:
                logging.info(len(meta_data_DF.loc[meta_data_DF['file'] == patient_to_take]))
                logging.info(patient_to_take)
                raise Exception('debug')

            meta_data_DF.loc[meta_data_DF['file'] == patient_to_take, 'use_in_balanced_dataset'] = 1

        meta_data_DF = meta_data_DF[meta_data_DF['use_in_balanced_dataset'] == 1]
        meta_data_DF.reset_index(inplace=True)

    else:
        meta_data_DF['use_in_balanced_data_ER'] = 0
        meta_data_DF.loc[meta_data_DF['ER status'] == 'Negative', 'use_in_balanced_data_ER'] = 1  # take all negatives

        # from all positives, take the same amount as negatives
        patient_list, patient_ind_list, patient_inverse_list = np.unique(
            np.array(meta_data_DF['patient barcode']).astype('str'), return_index=True, return_inverse=True)

        # get patient status for each patient
        # for patients with multiple statuses, the first one will be taken. These cases are rare.
        patient_status = []
        for i_patient in patient_ind_list:
            patient_status.append(meta_data_DF.loc[i_patient, 'ER status'])

        N_negative_patients = np.sum(np.array(patient_status) == 'Negative')
        positive_patient_ind_list = np.where(np.array(patient_status) == 'Positive')

        # take N_negative_patients positive patient
        positive_patients_inds_to_take = sample(list(positive_patient_ind_list[0]), k=N_negative_patients)
        for patient_to_take in positive_patients_inds_to_take:
            meta_data_DF.loc[patient_inverse_list == patient_to_take, 'use_in_balanced_data_ER'] = 1

    return meta_data_DF


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        w = weight if weight is not None else torch.FloatTensor([1., 1.])
        self.register_buffer("weight", w)
        self.gamma = gamma

    def forward(self, input, target):
        ce = F.cross_entropy(input, target.long(), reduction='none')
        pt = torch.exp(-ce)
        ce *= torch.matmul(torch.nn.functional.one_hot(target.long(), num_classes=2).float(), self.weight)
        return ((1 - pt) ** self.gamma * ce).mean()


class EmbedSquare(object):
    def __init__(self, size=16, stride=8, pad=4, minibatch_size=1, color='Testing'):
        self.size = size
        self.stride = stride
        self.pad = pad
        self.minibatch = minibatch_size
        self.normalized_square = torch.zeros(1, 3, self.size, self.size)
        self.color = color

        if color == 'Black':
            self.normalized_square[:, 0, :, :], \
            self.normalized_square[:, 1, :, :], \
            self.normalized_square[:, 2, :, :] = -7.9982, -4.7133, -11.8895  # Value of BLACK pixel after normalization
        elif color == 'White':
            self.normalized_square[:, 0, :, :], \
            self.normalized_square[:, 1, :, :], \
            self.normalized_square[:, 2, :, :] = 0.8907, 0.9977, 0.8170  # Value of WHITE pixel after normalization
        elif color == 'Gray':
            self.normalized_square[:, 0, :, :], \
            self.normalized_square[:, 1, :, :], \
            self.normalized_square[:, 2, :, :] = -3.5712, - 1.8690, - 5.5611  # Value of GRAY pixel after normalization
        elif color == 'Testing':
            self.normalized_square = torch.ones(1, 3, self.size, self.size) * 255
        else:
            raise Exception('Color choice do not match to any of the options (White/ Black)')

    def __call__(self, image):
        if len(image.shape) == 3:
            _, tile_size, _ = image.shape
        elif len(image.shape) == 4:
            _, _, tile_size, _ = image.shape

        if type(image) != torch.Tensor:
            raise Exception('Image is not in correct format')

        new_image = torch.zeros((1, 3, tile_size + 2 * self.pad, tile_size + 2 * self.pad))
        new_image[:, :, self.pad:self.pad + tile_size, self.pad:self.pad + tile_size] = image

        init = {'Row': 0,
                'Col': 0}

        total_jumps = 256 // self.stride
        output_images = [image.reshape([1, 3, tile_size, tile_size])]
        minibatch_size = 0
        counter = 0
        image_output_minibatch = torch.zeros((self.minibatch, 3, tile_size, tile_size))
        if tile_size == 2048:  # We'll need to put 16 squares (4X4) and not just one
            # At first we need to create a basic square mask.
            basic_mask = np.zeros((1808, 1808), dtype=bool)
            for row_idx in range(8):
                for col_idx in range(8):
                    basic_mask[256 * row_idx:256 * row_idx + 16, 256 * col_idx:256 * col_idx + 16] = True

        logging.info('Creating square embeded tiles...')
        for row_idx in tqdm(range(0, total_jumps)):
            init['Row'] = row_idx * self.stride
            init['Col'] = 0
            for col_idx in range(0, total_jumps):
                image_output = torch.clone(new_image)
                init['Col'] = col_idx * self.stride
                if tile_size == 256:
                    image_output[:, :, init['Row']:self.size + init['Row'],
                    init['Col']:self.size + init['Col']] = self.normalized_square

                elif tile_size == 2048:  # We need to put 16 squares (4X4) and not just one
                    mask = np.zeros_like(image_output, dtype=bool)
                    mask[:, :, init['Row']:1808 + init['Row'], init['Col']:1808 + init['Col']] = basic_mask
                    np.place(image_output.numpy(), mask, self.normalized_square[0, 0, 0, 0].item())

                # Now we have to cut the padding:
                image_output = image_output[:, :, self.pad: + self.pad + tile_size, self.pad: + self.pad + tile_size]

                # And add the output image to the list of ready images with the cutout:
                image_output_minibatch[minibatch_size, :, :, :] = image_output
                minibatch_size += 1
                counter += 1

                if counter == 1024:
                    output_images.append(image_output_minibatch[:minibatch_size, :, :, :])
                    break
                if minibatch_size == self.minibatch:
                    output_images.append(image_output_minibatch)
                    minibatch_size = 0
                    image_output_minibatch = torch.zeros((self.minibatch, 3, tile_size, tile_size))

        return output_images


def get_label(target, multi_target=False):
    if multi_target:
        label = []
        for t in target:
            label.append(get_label(t))
        return label
    else:
        if target == 'Positive':
            return [1]
        elif target == 'Negative':
            return [0]
        elif ((isinstance(target, int) or isinstance(target, float)) and not np.isnan(target)) or (
                (target.__class__ == str) and (target.isnumeric())):  # support multiclass
            return [int(target)]
        else:  # unknown
            return [-1]


def num_2_bool(num):
    if num == 1:
        return True
    elif num == 0:
        return False
    else:
        return -1


def plot_grad_flow(named_parameters):
    # taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063
    """ Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    from matplotlib.lines import Line2D
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(max_grads) * 1.05)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.gcf().subplots_adjust(bottom=0.5)


def send_run_data_via_mail():
    if sys.platform != 'linux':  # Send mail only when running on server
        return
    else:
        import yagmail

    path_parts = os.getcwd().split('/')
    if 'womer' in path_parts:
        filename = '/home/womer/project/runs/run_data.xlsx'
        user = 'womer'

    elif 'rschley' in path_parts:
        filename = '/home/rschley/code/WSI_MIL/WSI_MIL/runs/run_data.xlsx'
        user = 'rschley'

    elif 'sgils' in path_parts:
        filename = '/mnt/gipnetapp_public/sgils/ran/runs/run_data.xlsx'
        user = 'sgils'

    elif 'talneoran' in path_parts:
        filename = '/home/talneoran/workspace/wsi/runs/run_data.xlsx'
        user = 'talneoran'

    elif 'dahen' in path_parts:
        filename = '/home/dahen/WSI_ran_legacy/WSI/runs/run_data.xlsx'
        user = 'dahen'

    elif 'shacharcohen' in path_parts:
        filename = '/home/shacharcohen/workspace/WSI/runs/run_data.xlsx'
        user = 'Shachar'

    else:
        logging.info('This user parameters are not defined. Email will not be sent')
        return

    yag = yagmail.SMTP('gipmed.python@gmail.com')
    yag.send(
        to='gipmed.python@gmail.com',
        subject=user + ': run_data.xlsx',
        attachments=filename,
    )

    logging.info('email sent to gipmed.python@gmail.com')


def cohort_to_int(cohort_list: list) -> list:
    CohortDictionary = {'ABCTB': 0,
                        'CARMEL1': 1,
                        'CARMEL2': 1,
                        'CARMEL3': 1,
                        'CARMEL4': 1,
                        'CARMEL5': 1,
                        'CARMEL6': 1,
                        'CARMEL7': 1,
                        'CARMEL8': 1,
                        'TCGA': 2,
                        'HAEMEK': 3,
                        'HAEMEK1': 3
                        }

    return [CohortDictionary[key] for key in cohort_list]


def start_log(args, to_file=False):
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    if to_file:
        logfile = os.path.join(args.output_dir, 'log.txt')
        os.makedirs(args.output_dir, exist_ok=True)
        handlers = [stream_handler,
                    logging.FileHandler(filename=logfile)]
    else:
        handlers = [stream_handler]
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        handlers=handlers)
    logging.info('*** START ARGS ***')
    for k, v in vars(args).items():
        logging.info('{}: {}'.format(k, v))
    logging.info('*** END ARGS ***')