import os
import socket
import sys
from enum import Enum
import re
import numpy as np
import pandas as pd
from shutil import copyfile
import datetime


def name_of_gipdeep_host_node():
    return socket.gethostname()


class dataset_group(Enum):
    CARMEL = 0
    HAEMEK = 1
    BENIGN = 2
    HER2 = 3
    TMA = 4
    ABCTB = 5
    TCGA = 6
    SHEBA = 7
    IPATIMUP = 8
    COVILHA = 9
    HEROHE = 10
    HAEMEK_ONCO = 11
    TCGA_LUNG = 12


def get_dataset_group(dataset: str):
    dataset_stripped = re.sub(r'[0-9_]+', '', dataset)
    #support specific datasets with underscore or number
    if dataset_stripped == 'HER':
        dataset_stripped = 'HER2'
    elif dataset_stripped == 'HAEMEKONCO':
        dataset_stripped = 'HAEMEK_ONCO'
    elif dataset_stripped == 'TCGALUNG':
        dataset_stripped = 'TCGA_LUNG'
    return dataset_group[dataset_stripped]


def get_dataset_batch_num(dataset: str):
    # support up to 100 batches
    if not dataset[-1].isdigit():
        return ''
    if dataset[-2].isdigit():
        return dataset[-2:]
    else:
        return dataset[-1]


def get_slides_data_file(main_data_dir, dataset_name, extension=''):
    return os.path.join(main_data_dir, dataset_name, 'slides_data_' + dataset_name + extension + '.xlsx')


def open_excel_file(excel_file):
    if os.path.isfile(excel_file):
        excel_file_df = pd.read_excel(excel_file, engine='openpyxl')
        return excel_file_df
    else:
        raise IOError('cannot find excel file at: ' + excel_file)


def save_df_to_excel(df, file):
    df.to_excel(file, index=False)


def get_dataset_group_batch_list(dataset_group):
    dir_dict = get_datasets_dir_dict(dataset_group.name)
    batch_list = [get_dataset_batch_num(dataset) for dataset in dir_dict.keys()]
    return batch_list


def get_dataset_group_num_batches(dataset_group):
    return len(get_dataset_group_batch_list(dataset_group))


def backup_all_dataset_group_metadata(dataset_group, extension):
    dir_dict = get_datasets_dir_dict(dataset_group.name)
    for dataset in dir_dict.items():
        slides_data_file = os.path.join(dataset[1], 'slides_data_' + dataset[0] + '.xlsx')
        backup_dataset_metadata(slides_data_file, extension)


def backup_dataset_metadata(metadata_file, extension):
    time_format = "%d%m%y_%H%M%S"
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), time_format)
    file_split = os.path.splitext(metadata_file)
    assert len(file_split) == 2, "invalid slides_data file path"
    backup_file = file_split[0] + extension + '_' + timestamp + file_split[1]
    copyfile(metadata_file, backup_file)


def merge_dataset_group_metadata(dataset_group):
    dir_dict = get_datasets_dir_dict(dataset_group.name)
    slides_data_merged = pd.DataFrame()
    for dataset in dir_dict.items():
        slides_data_file = os.path.join(dataset[1], 'slides_data_' + dataset[0] + '.xlsx')

        slides_data = open_excel_file(slides_data_file)
        slides_data_merged = pd.concat([slides_data_merged, slides_data])
    return slides_data_merged


def unmerge_dataset_group_data(slides_data, dataset_group):
    dir_dict = get_datasets_dir_dict(dataset_group.name)
    for dataset in dir_dict.items():
        dataset_metadata = slides_data[slides_data['id'] == dataset[0]]
        slides_data_file = os.path.join(dataset[1], 'slides_data_' + dataset[0] + '.xlsx')
        save_df_to_excel(dataset_metadata, slides_data_file)


def get_datasets_dir_dict(Dataset: str):
    gipdeep10_used = name_of_gipdeep_host_node() == "gipdeep10"
    if gipdeep10_used:
        path_init = r'/data/'
    else:
        path_init = r'/mnt/gipmed_new/Data/'

    dir_dict = {}
    TCGA_gipdeep_path = path_init + r'Breast/TCGA'
    ABCTB_gipdeep_path = path_init + r'Breast/ABCTB_ndpi/ABCTB'
    HEROHE_gipdeep_path = path_init + r'Breast/HEROHE'
    SHEBA_gipdeep_path = path_init + r'Breast/Sheba'
    ABCTB_TIF_gipdeep_path = path_init + r'Breast/ABCTB_TIF'
    CARMEL_gipdeep_path = path_init + r'Breast/Carmel'
    TCGA_LUNG_gipdeep_path = path_init + r'Lung/TCGA_Lung/TCGA_LUNG'
    ALL_gipdeep_path = path_init + r'BoneMarrow/ALL'
    AML_gipdeep_path = path_init + r'BoneMarrow/AML/AML'
    Ipatimup_gipdeep_path = path_init + r'Breast/Ipatimup'
    Covilha_gipdeep_path = path_init + r'Breast/Covilha'
    TMA_HE_02_008_gipdeep_path = path_init + r'Breast/TMA/bliss_data/02-008/HE/TMA_HE_02-008'
    TMA_HE_01_011_gipdeep_path = path_init + r'Breast/TMA/bliss_data/01-011/HE/TMA_HE_01-011'
    HAEMEK_gipdeep_path = path_init + r'Breast/Haemek'
    CARMEL_BENIGN_gipdeep_path = path_init + r'Breast/Carmel/Benign'

    TCGA_ran_path = r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat\TCGA'
    HEROHE_ran_path = r'C:\ran_data\HEROHE_examples'
    ABCTB_ran_path = r'C:\ran_data\ABCTB\ABCTB_examples\ABCTB'
    TMA_HE_02_008_ran_path = r'C:\ran_data\TMA\02-008\TMA'
    SHEBA_ran_path = r'C:\ran_data\Sheba'
    ALL_ran_path = r'C:\ran_data\BoneMarrow'

    TCGA_omer_path = r'/Users/wasserman/Developer/WSI_MIL/All Data/TCGA'
    HEROHE_omer_path = r'/Users/wasserman/Developer/WSI_MIL/All Data/HEROHE'
    CARMEL_omer_path = r'/Users/wasserman/Developer/WSI_MIL/All Data/CARMEL'

    if Dataset == 'ABCTB_TCGA':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['TCGA'] = TCGA_gipdeep_path
            dir_dict['ABCTB'] = ABCTB_TIF_gipdeep_path
        elif sys.platform == 'win32':  # GIPdeep
            dir_dict['TCGA'] = TCGA_ran_path
            dir_dict['ABCTB'] = ABCTB_ran_path

    elif Dataset == 'CARMEL':
        if sys.platform == 'linux':  # GIPdeep
            for ii in np.arange(1, 9):
                dir_dict['CARMEL' + str(ii)] = os.path.join(CARMEL_gipdeep_path, '1-8', 'Batch_' + str(ii), 'CARMEL' + str(ii))
        elif sys.platform == 'darwin':  # Omer
            dir_dict['CARMEL'] = CARMEL_omer_path

    elif Dataset == 'CARMEL+BENIGN':
        if sys.platform == 'linux':  # GIPdeep
            for ii in np.arange(1, 9):
                dir_dict['CARMEL' + str(ii)] = os.path.join(CARMEL_gipdeep_path, '1-8', 'Batch_' + str(ii), 'CARMEL' + str(ii))

            for ii in np.arange(1, 4):
                dir_dict['BENIGN' + str(ii)] = os.path.join(CARMEL_BENIGN_gipdeep_path, 'Batch_' + str(ii), 'BENIGN' + str(ii))

    elif Dataset == 'Carmel 9-11':
        if sys.platform == 'linux':  # GIPdeep
            for ii in np.arange(9, 12):
                dir_dict['CARMEL' + str(ii)] = os.path.join(CARMEL_gipdeep_path, '9-11', 'Batch_' + str(ii), 'CARMEL' + str(ii))
        elif sys.platform == 'darwin':  # Omer
            raise Exception('Need to implement')

    elif (Dataset[:6] == 'CARMEL') and (len(Dataset) > 6):
        batch_num = Dataset[6:]
        if sys.platform == 'linux':  # GIPdeep
            dir_dict[Dataset] = os.path.join(CARMEL_gipdeep_path, '1-8' if int(batch_num) < 9 else '9-11', 'Batch_' + batch_num, 'CARMEL' + batch_num)
        elif sys.platform == 'win32':  # Ran local
            dir_dict[Dataset] = TCGA_ran_path #temp for debug only

    elif (Dataset[:6] == 'BENIGN') and (len(Dataset) > 6):
        batch_num = Dataset[6:]
        if sys.platform == 'linux':  # GIPdeep
            dir_dict[Dataset] = os.path.join(CARMEL_BENIGN_gipdeep_path, 'Batch_' + batch_num, 'BENIGN' + batch_num)

    elif Dataset == 'CAT':
        if sys.platform == 'linux':  # GIPdeep
            for ii in np.arange(1, 9):
                dir_dict['CARMEL' + str(ii)] = os.path.join(CARMEL_gipdeep_path, '1-8', 'Batch_' + str(ii), 'CARMEL' + str(ii))
            dir_dict['TCGA'] = TCGA_gipdeep_path
            dir_dict['ABCTB'] = ABCTB_TIF_gipdeep_path

        elif sys.platform == 'win32':  #Ran local
            dir_dict['TCGA'] = TCGA_ran_path
            dir_dict['HEROHE'] = HEROHE_ran_path

        elif sys.platform == 'darwin':   #Omer local
            dir_dict['TCGA'] = TCGA_omer_path
            dir_dict['CARMEL'] = CARMEL_omer_path

        else:
            raise Exception('Unrecognized platform')

    elif Dataset == 'TCGA_LUNG':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['TCGA_LUNG'] = TCGA_LUNG_gipdeep_path

    elif Dataset == 'TCGA':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['TCGA'] = TCGA_gipdeep_path

        elif sys.platform == 'win32':  # Ran local
            dir_dict['TCGA'] = TCGA_ran_path

        elif sys.platform == 'darwin':  # Omer local
            dir_dict['TCGA'] = TCGA_omer_path

        else:
            raise Exception('Unrecognized platform')

    elif Dataset == 'HEROHE':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['HEROHE'] = HEROHE_gipdeep_path

        elif sys.platform == 'win32':  # Ran local
            dir_dict['HEROHE'] = HEROHE_ran_path

        elif sys.platform == 'darwin':  # Omer local
            dir_dict['HEROHE'] = HEROHE_omer_path

    elif Dataset == 'ABCTB_TIF':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['ABCTB_TIF'] = ABCTB_TIF_gipdeep_path
        elif sys.platform == 'darwin':  # Omer local
            dir_dict['ABCTB_TIF'] = r'All Data/ABCTB_TIF'
        else:
            raise Exception('Unsupported platform')

    elif Dataset == 'ABCTB_TILES':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['ABCTB_TILES'] = r'/home/womer/project/All Data/ABCTB_TILES'
        elif sys.platform == 'darwin':  # Omer local
            dir_dict['ABCTB_TILES'] = r'All Data/ABCTB_TILES'

    elif Dataset == 'ABCTB':
        if sys.platform == 'linux':  # GIPdeep Run from local files
            dir_dict['ABCTB'] = ABCTB_TIF_gipdeep_path

        elif sys.platform == 'win32':  # Ran local
            dir_dict['ABCTB'] = ABCTB_ran_path

        elif sys.platform == 'darwin':  # Omer local
            dir_dict['ABCTB'] = r'All Data/ABCTB_TIF'

    elif Dataset == 'SHEBA':
        if sys.platform == 'linux':
            for ii in np.arange(2, 7):
                dir_dict['SHEBA' + str(ii)] = os.path.join(SHEBA_gipdeep_path, 'Batch_' + str(ii), 'SHEBA' + str(ii))
        elif sys.platform == 'win32':  # Ran local
            for ii in np.arange(2, 7):
                dir_dict['SHEBA' + str(ii)] = os.path.join(SHEBA_ran_path, 'SHEBA' + str(ii))

    elif Dataset == 'PORTO_HE':
        if sys.platform == 'linux':
            dir_dict['PORTO_HE'] = r'/mnt/gipmed_new/Data/Lung/PORTO_HE'
        elif sys.platform == 'win32':  # Ran local
            dir_dict['PORTO_HE'] = r'C:\ran_data\Lung_examples\LUNG'
        elif sys.platform == 'darwin':  # Omer local
            dir_dict['PORTO_HE'] = 'All Data/LUNG'

    elif Dataset == 'PORTO_PDL1':
        if sys.platform == 'linux':
            dir_dict['PORTO_PDL1'] = r'/mnt/gipmed_new/Data/Lung/sgils/LUNG/PORTO_PDL1'
        elif sys.platform == 'win32':  # Ran local
            dir_dict['PORTO_PDL1'] = r'C:\ran_data\IHC_examples\PORTO_PDL1'

    elif Dataset == 'LEUKEMIA':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['ALL'] = ALL_gipdeep_path
            dir_dict['AML'] = AML_gipdeep_path

    elif Dataset == 'AML':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['AML'] = ALL_gipdeep_path

    elif Dataset == 'ALL':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['ALL'] = ALL_gipdeep_path
        elif sys.platform == 'win32':
            dir_dict['ALL'] = ALL_ran_path

    elif Dataset == 'IC':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['Ipatimup'] = Ipatimup_gipdeep_path
            dir_dict['Covilha'] = Covilha_gipdeep_path

    elif Dataset == 'HIC':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['Ipatimup'] = Ipatimup_gipdeep_path
            dir_dict['Covilha'] = Covilha_gipdeep_path
            dir_dict['HEROHE'] = HEROHE_gipdeep_path

    elif Dataset == 'TMA_HE_02_008':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['TMA_HE_02_008'] = TMA_HE_02_008_gipdeep_path
        else:
            dir_dict['TMA_HE_02_008'] = TMA_HE_02_008_ran_path

    elif Dataset == 'TMA_HE_01_011':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['TMA_HE_01_011'] = TMA_HE_01_011_gipdeep_path

    elif Dataset == 'HAEMEK':
        if sys.platform == 'linux':  # GIPdeep
            for ii in np.arange(1, 2):
                dir_dict['HAEMEK' + str(ii)] = os.path.join(HAEMEK_gipdeep_path, 'Batch_' + str(ii), 'HAEMEK' + str(ii))

    return dir_dict


def add_folder(main_folder, new_folder):
    new_path = os.path.join(main_folder, new_folder)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print('cannot delete ' + file_path + ': path does not exist')


def load_backup_slides_data(in_dir, dataset, extension=''):
    slides_data_file = get_slides_data_file(in_dir, dataset)
    if extension != '':
        backup_dataset_metadata(slides_data_file, extension=extension)
    slides_data_DF = open_excel_file(slides_data_file)
    return slides_data_file, slides_data_DF


def format_empty_spaces_as_string(workbook, worksheet, ind, column_list):
    text_format = workbook.add_format({'num_format': '@'})
    for col in column_list:
        worksheet.write(col + str(ind + 2), '', text_format)
    return worksheet


def is_mrxs(filename):
    return filename.split('.')[-1] == 'mrxs'