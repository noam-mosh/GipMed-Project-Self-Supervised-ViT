import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def get_RegModel_Features_location_dict(train_DataSet: str, target: str,
                                        test_fold: int):
    test_fold = test_fold if test_fold>0 else None
    All_Data_Dict = {
        'linux': {
            'CAT': {
                'Fold None' : { # TODO: wrangle this nightmare format.
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_40015-ER-TestFold_-1',
                        'TrainSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40015-ER-TestFold_-1/Inference/train_w_features',
                        'TestSet Location': None,
                        'REG Model Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40015-ER-TestFold_-1/Model_CheckPoints/model_data_Last_Epoch.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_40019-Her2-TestFold_-1',
                        'TrainSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Inference/train_w_features',
                        'TestSet Location': None,
                        'REG Model Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Model_CheckPoints/model_data_Last_Epoch.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_40046-PR-TestFold_-1',
                        'TrainSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40046-PR-TestFold_-1/Inference/train_w_features',
                        'TestSet Location': None,
                        'REG Model Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40046-PR-TestFold_-1/Model_CheckPoints/model_data_Last_Epoch.pt'
                    }
                },
                'Fold 1': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_355-ER-TestFold_1 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/train_w_features_w_locs',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/test_w_features_locs',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_392-Her2-TestFold_1 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/test_w_locs_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_10-PR-TestFold_1  With Locations',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/train_w_features_new',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/test_w_locs_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                },
                'Fold 2': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_393-ER-TestFold_2 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/train_w_features_new',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_412-Her2-TestFold_2 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20063-PR-TestFold_2 With Locations',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/train_w_features_new',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                },
                'Fold 3': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_472-ER-TestFold_3',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_20114-Her2-TestFold_3',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/train_w_features',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_497-PR-TestFold_3',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 4': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_542-ER-TestFold_4',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_20201-Her2-TestFold_4',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/train_w_features',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20207-PR-TestFold_4',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/train_w_features',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 5': {
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_20228-Her2-TestFold_5',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/train_w_features',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_611-ER-TestFold_5',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_611-ER-TestFold_5/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_611-ER-TestFold_5/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_611-ER-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20235-PR-TestFold_5',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20235-PR-TestFold_5/Inference/train_w_features',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20235-PR-TestFold_5/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20235-PR-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                }
            },
            'CARMEL': {
                'Fold None' : {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_40022-ER-TestFold_-1',
                        'TrainSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40022-ER-TestFold_-1/Inference/train_w_features',
                        'TestSet Location': None,
                        'REG Model Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40022-ER-TestFold_-1/Model_CheckPoints/model_data_Last_Epoch.pt'
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_40023-Ki67-TestFold_-1',
                        'TrainSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40023-Ki67-TestFold_-1/Inference/train_w_features',
                        'TestSet Location': None,
                        'REG Model Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40023-Ki67-TestFold_-1/Model_CheckPoints/model_data_Last_Epoch.pt'
                    },
                    'Her2': {
                        'DataSet Name': None,
                        'TrainSet Location': None,
                        'TestSet Location': None,
                        'REG Model Location': None
                    },
                    'PR': {
                        'DataSet Name': None,
                        'TrainSet Location': None,
                        'TestSet Location': None,
                        'REG Model Location': None
                    }
                },
                'Fold 1': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_358-ER-TestFold_1',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_419-Ki67-TestFold_1',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name': None,
                        'TrainSet Location': None,
                        'TestSet Location': None,
                        'REG Model Location': None
                    },
                    'PR': {
                        'DataSet Name': None,
                        'TrainSet Location': None,
                        'TestSet Location': None,
                        'REG Model Location': None
                    }
                },
                'Fold 2': {
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_490-Ki67-TestFold_2',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 3': {
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_20152-Ki67-TestFold_3',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20152-Ki67-TestFold_3/Inference/train_w_features',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20152-Ki67-TestFold_3/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20152-Ki67-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 4': {
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_30090-Ki67-TestFold_4',
                        'TrainSet Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30090-Ki67-TestFold_4/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30090-Ki67-TestFold_4/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30090-Ki67-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 5': {
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_30092-Ki67-TestFold_5',
                        'TrainSet Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30092-Ki67-TestFold_5/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30092-Ki67-TestFold_5/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30092-Ki67-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                }
            },
            'CARMEL 9-11': {
                'Fold None' : {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_40015-ER-TestFold_-1, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40015-ER-TestFold_-1/Inference/CARMEL9',
                            'Carmel 10':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40015-ER-TestFold_-1/Inference/CARMEL10',
                            'Carmel 11':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40015-ER-TestFold_-1/Inference/CARMEL11'
                        }
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_40046-PR-TestFold_-1, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40046-PR-TestFold_-1/Inference/CARMEL9',
                            'Carmel 10':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40046-PR-TestFold_-1/Inference/CARMEL10',
                            'Carmel 11':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40046-PR-TestFold_-1/Inference/CARMEL11'
                        }
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_40019-Her2-TestFold_-1, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Inference/CARMEL9',
                            'Carmel 10':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Inference/CARMEL10',
                            'Carmel 11':
                            '/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Inference/CARMEL11'
                        }
                    }
                },
                'Fold 1': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_355-ER-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/CARMEL9',
                            'Carmel 10':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/CARMEL10',
                            'Carmel 11':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/CARMEL11'
                        }
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_419-Ki67-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/CARMEL9/',
                            'Carmel 10':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/CARMEL10/',
                            'Carmel 11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/CARMEL11/'
                        }
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_392-Her2-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            "/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/CARMEL9",
                            'Carmel 10':
                            "/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/CARMEL10",
                            'Carmel 11':
                            "/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/CARMEL11",
                        }
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_10-PR-TestFold_1, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/CARMEL9',
                            'Carmel 10':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/CARMEL10',
                            'Carmel 11':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/CARMEL11'
                        }
                    },
                },
                'Fold 2': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_393-ER-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/CARMEL9',
                            'Carmel 10':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/CARMEL10',
                            'Carmel 11':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/CARMEL11'
                        }
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_490-Ki67-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/CARMEL9/',
                            'Carmel 10':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/CARMEL10/',
                            'Carmel 11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/CARMEL11/'
                        }
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_412-Her2-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            "/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/CARMEL9",
                            'Carmel 10':
                            "/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/CARMEL10",
                            'Carmel 11':
                            "/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/CARMEL11",
                        }
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_20063-PR-TestFold_2, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/CARMEL9',
                            'Carmel 10':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/CARMEL10',
                            'Carmel 11':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/CARMEL11'
                        }
                    },
                },
                'Fold 3': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_472-ER-TestFold_3, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/CARMEL9',
                            'Carmel 10':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/CARMEL10',
                            'Carmel 11':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/CARMEL11'
                        }
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_20114-Her2-TestFold_3, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            "/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/CARMEL9",
                            'Carmel 10':
                            "/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/CARMEL10",
                            'Carmel 11':
                            "/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/CARMEL11",
                        }
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_20152-Ki67-TestFold_3, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20152-Ki67-TestFold_3/Inference/CARMEL9/',
                            'Carmel 10':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20152-Ki67-TestFold_3/Inference/CARMEL10/',
                            'Carmel 11':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20152-Ki67-TestFold_3/Inference/CARMEL11/'
                        }
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_497-PR-TestFold_3, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/CARMEL9',
                            'Carmel 10':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/CARMEL10',
                            'Carmel 11':
                            '/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/CARMEL11'
                        }
                    },
                },
                'Fold 4': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_542-ER-TestFold_4, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/CARMEL9',
                            'Carmel 10':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/CARMEL10',
                            'Carmel 11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/CARMEL11'
                        }
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_20201-Her2-TestFold_4, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/CARMEL9',
                            'Carmel 10':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/CARMEL10',
                            'Carmel 11':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/CARMEL11'
                        }
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_20207-PR-TestFold_4, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/CARMEL9',
                            'Carmel 10':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/CARMEL10',
                            'Carmel 11':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/CARMEL11'
                        }
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_30090-Ki67-TestFold_4, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/home/talneoran/workspace/wsi/runs/Exp_30090-Ki67-TestFold_4/Inference/CARMEL9',
                            'Carmel 10':
                            r'/home/talneoran/workspace/wsi/runs/Exp_30090-Ki67-TestFold_4/Inference/CARMEL10',
                            'Carmel 11':
                            r'/home/talneoran/workspace/wsi/runs/Exp_30090-Ki67-TestFold_4/Inference/CARMEL11',
                        }
                    }
                },
                'Fold 5': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_611-ER-TestFold_5, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_611-ER-TestFold_5/Inference/CARMEL9',
                            'Carmel 10':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_611-ER-TestFold_5/Inference/CARMEL10',
                            'Carmel 11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_611-ER-TestFold_5/Inference/CARMEL11'
                        }
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_20228-Her2-TestFold_5, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/CARMEL9',
                            'Carmel 10':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/CARMEL10',
                            'Carmel 11':
                            r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/CARMEL11'
                        }
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Model From Exp_20235-PR-TestFold_5, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20235-PR-TestFold_5/Inference/CARMEL9',
                            'Carmel 10':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20235-PR-TestFold_5/Inference/CARMEL10',
                            'Carmel 11':
                            '/mnt/gipnetapp_public/sgils/ran/runs/Exp_20235-PR-TestFold_5/Inference/CARMEL11'
                        }
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_30092-Ki67-TestFold_5, CARMEL ONLY Slides Batch 9-11',
                        'TestSet Location': {
                            'Carmel 9':
                            r'/home/talneoran/workspace/wsi/runs/Exp_30092-Ki67-TestFold_5/Inference/CARMEL9',
                            'Carmel 10':
                            r'/home/talneoran/workspace/wsi/runs/Exp_30092-Ki67-TestFold_5/Inference/CARMEL10',
                            'Carmel 11':
                            r'/home/talneoran/workspace/wsi/runs/Exp_30092-Ki67-TestFold_5/Inference/CARMEL11',
                        }
                    }
                },
            },
            'HAEMEK': {
                'Fold None': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_40015-ER-TestFold_-1, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40015-ER-TestFold_-1/Inference/HAEMEK'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_40046-PR-TestFold_-1, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40046-PR-TestFold_-1/Inference/HAEMEK'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_40019-Her2-TestFold_-1, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40019-Her2-TestFold_-1/Inference/HAEMEK'
                    },
                },
                'Fold 1': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_355-ER-TestFold_1, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/HAEMEK'
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_419-Ki67-TestFold_1, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_419-Ki67-TestFold_1/Inference/HAEMEK'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_392-Her2-TestFold_1, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/HAEMEK'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20010-PR-TestFold_1, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/HAEMEK'
                    }
                },
                'Fold 2': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_393-ER-TestFold_2, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/HAEMEK',
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_490-Ki67-TestFold_2, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_490-Ki67-TestFold_2/Inference/HAEMEK',
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_412-Her2-TestFold_2, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r"/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/HAEMEK",
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20063-PR-TestFold_2, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/HAEMEK',
                    },
                },
                'Fold 3': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_472-ER-TestFold_3, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_472-ER-TestFold_3/Inference/HAEMEK'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_20114-Her2-TestFold_3, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r"/mnt/gipnetapp_public/sgils/ran/runs/Exp_20114-Her2-TestFold_3/Inference/HAEMEK"
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_20152-Ki67-TestFold_3, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20152-Ki67-TestFold_3/Inference/HAEMEK'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_497-PR-TestFold_3, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_497-PR-TestFold_3/Inference/HAEMEK'
                    },
                },
                'Fold 4': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_542-ER-TestFold_4, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_542-ER-TestFold_4/Inference/HAEMEK'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_20201-Her2-TestFold_4, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20201-Her2-TestFold_4/Inference/HAEMEK'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20207-PR-TestFold_4, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20207-PR-TestFold_4/Inference/HAEMEK'
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_30090-Ki67-TestFold_4, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30090-Ki67-TestFold_4/Inference/HAEMEK',
                    }
                },
                'Fold 5': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_611-ER-TestFold_5, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_611-ER-TestFold_5/Inference/HAEMEK'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_20228-Her2-TestFold_5, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20228-Her2-TestFold_5/Inference/HAEMEK'
                    },
                    'Ki67': {
                        'DataSet Name':
                        r'FEATURES: Exp_30092-Ki67-TestFold_5, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/home/talneoran/workspace/wsi/runs/Exp_30092-Ki67-TestFold_5/Inference/HAEMEK',
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20235-PR-TestFold_5, HAEMEK ONLY Slides',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20235-PR-TestFold_5/Inference/HAEMEK'
                    }
                },
            },
            'TCGA_ABCTB->CARMEL': {
                'Fold None': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40021-ER-TestFold_-1',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40021-ER-TestFold_-1/Inference/CARMEL1-8',
                            'CARMEL9':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40021-ER-TestFold_-1/Inference/CARMEL9',
                            'CARMEL10':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40021-ER-TestFold_-1/Inference/CARMEL10',
                            'CARMEL11':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40021-ER-TestFold_-1/Inference/CARMEL11',
                        }
                    }
                },
                'Fold 1': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_293-ER-TestFold_1',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/CARMEL9-11'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_308-Her2-TestFold_1',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Inference/CARMEL9-11'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_309-PR-TestFold_1',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Inference/CARMEL9-11'
                        }
                    }
                },
                'Fold 2': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_299-ER-TestFold_2',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_299-ER-TestFold_2/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_299-ER-TestFold_2/Inference/CARMEL9-11'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_310-Her2-TestFold_2',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_310-Her2-TestFold_2/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_310-Her2-TestFold_2/Inference/CARMEL9-11'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_320-PR-TestFold_2',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_320-PR-TestFold_2/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_320-PR-TestFold_2/Inference/CARMEL9-11'
                        }
                    }
                },
                'Fold 3': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_301-ER-TestFold_3',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_301-ER-TestFold_3/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_301-ER-TestFold_3/Inference/CARMEL9-11'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_311-Her2-TestFold_3',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_311-Her2-TestFold_3/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_311-Her2-TestFold_3/Inference/CARMEL9-11'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_312-PR-TestFold_3',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_312-PR-TestFold_3/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_312-PR-TestFold_3/Inference/CARMEL9-11'
                        }
                    }
                },
                'Fold 4': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_302-ER-TestFold_4',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_302-ER-TestFold_4/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_302-ER-TestFold_4/Inference/CARMEL9-11'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_334-Her2-TestFold_4',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_334-Her2-TestFold_4/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_334-Her2-TestFold_4/Inference/CARMEL9-11'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_313-PR-TestFold_4',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_313-PR-TestFold_4/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_313-PR-TestFold_4/Inference/CARMEL9-11'
                        }
                    }
                },
                'Fold 5': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_303-ER-TestFold_5',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_303-ER-TestFold_5/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_303-ER-TestFold_5/Inference/CARMEL9-11'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_348-Her2-TestFold_5',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_348-Her2-TestFold_5/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_348-Her2-TestFold_5/Inference/CARMEL9-11'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_321-PR-TestFold_5',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_321-PR-TestFold_5/Inference/CARMEL1-8',
                            'CARMEL 9-11':
                            r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_321-PR-TestFold_5/Inference/CARMEL9-11'
                        }
                    }
                },
            },
            'CARMEL->CARMEL 9-11': {
                'Fold None': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40022-ER-TestFold_-1',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40022-ER-TestFold_-1/Inference/CARMEL1-8',
                            'CARMEL9':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40022-ER-TestFold_-1/Inference/CARMEL9',
                            'CARMEL10':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40022-ER-TestFold_-1/Inference/CARMEL10',
                            'CARMEL11':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40022-ER-TestFold_-1/Inference/CARMEL11',
                        }
                    },
                    'Ki67': {
                        'DataSet Name': r'FEATURES: Exp_40023-Ki67-TestFold_-1',
                        'TestSet Location': {
                            'CARMEL':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40023-Ki67-TestFold_-1/Inference/CARMEL1-8',
                            'CARMEL9':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40023-Ki67-TestFold_-1/Inference/CARMEL9',
                            'CARMEL10':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40023-Ki67-TestFold_-1/Inference/CARMEL10',
                            'CARMEL11':
                            r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40023-Ki67-TestFold_-1/Inference/CARMEL11',
                        }
                    }
                }
            },
            'CARMEL->HAEMEK': {
                'Fold None': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40022-ER-TestFold_-1',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40022-ER-TestFold_-1/Inference/HAEMEK'
                        }
                    },
                    'Ki67': {
                        'DataSet Name': r'FEATURES: Exp_40023-Ki67-TestFold_-1',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40023-Ki67-TestFold_-1/Inference/HAEMEK'
                        }
                    }
                }
            },
            'TCGA_ABCTB->HAEMEK': {
                'Fold None': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40021-ER-TestFold_-1',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40021-ER-TestFold_-1/Inference/HAEMEK'
                        }
                    }
                },
                'Fold 1': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40005-ER-TestFold_1',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40005-ER-TestFold_1/Inference/HAEMEK'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_308-Her2-TestFold_1',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Inference/HAEMEK'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_309-PR-TestFold_1',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Inference/HAEMEK'
                        }
                    }
                },
                'Fold 2': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40006-ER-TestFold_2',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40006-ER-TestFold_2/Inference/HAEMEK'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_310-Her2-TestFold_2',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_310-Her2-TestFold_2/Inference/HAEMEK'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_320-PR-TestFold_2',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_320-PR-TestFold_2/Inference/HAEMEK'
                        }
                    }
                },
                'Fold 3': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40007-ER-TestFold_3',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40007-ER-TestFold_3/Inference/HAEMEK'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_311-Her2-TestFold_3',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_311-Her2-TestFold_3/Inference/HAEMEK'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_312-PR-TestFold_3',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_312-PR-TestFold_3/Inference/HAEMEK'
                        }
                    }
                },
                'Fold 4': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40008-ER-TestFold_4',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40008-ER-TestFold_4/Inference/HAEMEK'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_334-Her2-TestFold_4',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_334-Her2-TestFold_4/Inference/HAEMEK'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_313-PR-TestFold_4',
                        'TestSet Location': {
                            'CARMEL':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_313-PR-TestFold_4/Inference/HAEMEK'
                        }
                    }
                },
                'Fold 5': {
                    'ER': {
                        'DataSet Name': r'FEATURES: Exp_40009-ER-TestFold_5',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40009-ER-TestFold_5/Inference/HAEMEK'
                        }
                    },
                    'Her2': {
                        'DataSet Name': r'FEATURES: Exp_348-Her2-TestFold_5',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_348-Her2-TestFold_5/Inference/HAEMEK'
                        }
                    },
                    'PR': {
                        'DataSet Name': r'FEATURES: Exp_321-PR-TestFold_5',
                        'TestSet Location': {
                            'HAEMEK':
                                r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_321-PR-TestFold_5/Inference/HAEMEK'
                        }
                    }
                },
            },
            'TCGA_ABCTB': {
                'Fold None': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_40021-ER-TestFold_-1',
                        'TrainSet Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40021-ER-TestFold_-1/Inference/train_w_features',
                        'TestSet Location': None,
                        'REG Model Location':
                        r'/home/dahen/WSI_ran_legacy/WSI/runs/Exp_40021-ER-TestFold_-1/Model_CheckPoints/model_data_Last_Epoch.pt'
                    }
                },
                'Fold 1': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_293-ER-TestFold_1',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/train_inference_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/test_inference_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_308-Her2-TestFold_1',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_308-Her2-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_309-PR-TestFold_1',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Inference/test_features_with_tiff_slides',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_309-PR-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 2': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_299-ER-TestFold_2',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_299-ER-TestFold_2/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_299-ER-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_299-ER-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_310-Her2-TestFold_2',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_310-Her2-TestFold_2/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_310-Her2-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_310-Her2-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_320-PR-TestFold_2',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_320-PR-TestFold_2/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_320-PR-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_320-PR-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 3': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_301-ER-TestFold_3',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_301-ER-TestFold_3/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_301-ER-TestFold_3/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_301-ER-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_311-Her2-TestFold_3',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_311-Her2-TestFold_3/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_311-Her2-TestFold_3/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_311-Her2-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_312-PR-TestFold_3',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_312-PR-TestFold_3/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_312-PR-TestFold_3/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_312-PR-TestFold_3/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 4': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_302-ER-TestFold_4',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_302-ER-TestFold_4/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_302-ER-TestFold_4/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_302-ER-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_334-Her2-TestFold_4',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_334-Her2-TestFold_4/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_334-Her2-TestFold_4/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_334-Her2-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_313-PR-TestFold_4',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_313-PR-TestFold_4/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_313-PR-TestFold_4/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_313-PR-TestFold_4/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
                'Fold 5': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_303-ER-TestFold_5',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_303-ER-TestFold_5/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_303-ER-TestFold_5/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_303-ER-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_348-Her2-TestFold_5',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_348-Her2-TestFold_5/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_348-Her2-TestFold_5/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_348-Her2-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_321-PR-TestFold_5',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_321-PR-TestFold_5/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_321-PR-TestFold_5/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_321-PR-TestFold_5/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                },
            },
            'CAT with Location': {
                'Fold 1': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_355-ER-TestFold_1 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/train_w_features_w_locs',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/test_w_features_locs',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_392-Her2-TestFold_1 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Inference/test_w_locs_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_392-Her2-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_10-PR-TestFold_1  With Locations',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/train_w_features_new',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Inference/test_w_locs_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_10-PR-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                },
                'Fold 2': {
                    'ER': {
                        'DataSet Name':
                        r'FEATURES: Exp_393-ER-TestFold_2 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/train_w_features_new',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_393-ER-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'Her2': {
                        'DataSet Name':
                        r'FEATURES: Exp_412-Her2-TestFold_2 With Locations',
                        'TrainSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/train_w_features',
                        'TestSet Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_412-Her2-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    },
                    'PR': {
                        'DataSet Name':
                        r'FEATURES: Exp_20063-PR-TestFold_2 With Locations',
                        'TrainSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/train_w_features_new',
                        'TestSet Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Inference/test_w_features',
                        'REG Model Location':
                        r'/mnt/gipnetapp_public/sgils/ran/runs/Exp_20063-PR-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                }
            }
        }
    }

    if '+is_Tumor' in target:
        receptor = target.split('+')[0]
        if receptor == 'Her2':
            return All_Data_Dict[sys.platform][train_DataSet][
                'Fold ' + str(test_fold)][receptor], All_Data_Dict[
                    sys.platform][train_DataSet][
                        'Fold ' + str(test_fold)]['is_Tumor_for_Her2']

        elif receptor == 'PR':
            return All_Data_Dict[sys.platform][train_DataSet][
                'Fold ' + str(test_fold)][receptor], All_Data_Dict[
                    sys.platform][train_DataSet][
                        'Fold ' + str(test_fold)]['is_Tumor_for_PR']

        elif receptor == 'ER':
            return All_Data_Dict[sys.platform][train_DataSet][
                'Fold ' + str(test_fold)]['ER_for_is_Tumor'], All_Data_Dict[
                    sys.platform][train_DataSet][
                        'Fold ' + str(test_fold)]['is_Tumor_for_ER']

    else:
        return All_Data_Dict[sys.platform][train_DataSet][
            'Fold ' + str(test_fold)][target]


def dataset_properties_to_location(dataset_name_list: list,
                                   receptor: str,
                                   test_fold: int,
                                   is_train: bool = False):
    # Basic data definition:
    if sys.platform == 'darwin':
        dataset_full_data_dict = {
            'TCGA_ABCTB': {
                'ER': {
                    1: {
                        'Train':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Train',
                        'Test':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Test',
                        'Dataset name': r'FEATURES: Exp_293-ER-TestFold_1'
                    }
                }
            },
            'CAT': {
                'ER': {
                    1: {
                        'Train':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Train',
                        'Test':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test',
                        'Dataset name':
                        r'FEATURES: Exp_355-ER-TestFold_1',
                        'Regular model location':
                        r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_355_TF_1/model_data_Epoch_1000.pt'
                    }
                }
            },
            'CARMEL': {
                'ER': {
                    1: {
                        'Train':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Train',
                        'Test':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Test',
                        'Dataset name':
                        r'FEATURES: Exp_358-ER-TestFold_1',
                        'Regular model location':
                        r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_358-TF_1/model_data_Epoch_1000.pt'
                    }
                }
            },
            'CARMEL_40': {
                'ER': {
                    1: {
                        'Train':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Train',
                        'Test':
                        r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Test',
                        'Dataset name':
                        r'FEATURES: Exp_381-ER-TestFold_1',
                        'Regular model location':
                        r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_381-TF_1/model_data_Epoch_1200.pt'
                    }
                }
            }
        }
    elif sys.platform == 'linux':
        dataset_full_data_dict = {
            'TCGA_ABCTB': {
                'ER': {
                    1: {
                        'Train':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/train_inference_w_features',
                        'Test':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_293-ER-TestFold_1/Inference/test_inference_w_features',
                        'Dataset name': r'FEATURES: Exp_293-ER-TestFold_1'
                    }
                }
            },
            'CAT': {
                'ER': {
                    1: {
                        'Train':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/train_w_features',
                        'Test':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Inference/test_w_features',
                        'Dataset name':
                        r'FEATURES: Exp_355-ER-TestFold_1',
                        'Regular model location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                }
            },
            'CARMEL': {
                'ER': {
                    1: {
                        'Train':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/train_w_features',
                        'Test':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Inference/test_w_features',
                        'Dataset name':
                        r'FEATURES: Exp_358-ER-TestFold_1',
                        'Regular model location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_358-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'
                    }
                }
            },
            'CARMEL_40': {
                'ER': {
                    1: {
                        'Train':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Inference/train_w_features',
                        'Test':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Inference/test_w_features',
                        'Dataset name':
                        r'FEATURES: Exp_358-ER-TestFold_1',
                        'Regular model location':
                        r'/home/rschley/code/WSI_MIL/WSI_MIL/runs/Exp_381-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1200.pt'
                    }
                }
            }
        }

    dataset_location_list = []

    if receptor == 'ER_Features':
        receptor = 'ER'
    for dataset in dataset_name_list:
        location = dataset_full_data_dict[dataset][receptor][test_fold][
            'Train' if is_train else 'Test']
        dataset_name = dataset_full_data_dict[dataset][receptor][test_fold][
            'Dataset name']
        regular_model_location = dataset_full_data_dict[dataset][receptor][
            test_fold]['Regular model location']
        dataset_location_list.append(
            [dataset, location, dataset_name, regular_model_location])

    return dataset_location_list


def save_all_slides_and_models_data(all_slides_tile_scores,
                                    all_slides_final_scores,
                                    all_slides_weights_before_softmax,
                                    all_slides_weights_after_softmax,
                                    models,
                                    Output_Dirs,
                                    Epochs,
                                    data_path,
                                    true_test_path: str = ''):
    # Save slide scores to file:
    for num_model in range(len(models)):
        if type(Output_Dirs) == str:
            output_dir = Output_Dirs
        elif type(Output_Dirs) is list:
            output_dir = Output_Dirs[num_model]

        epoch = Epochs[num_model]
        model = models[num_model]

        full_output_dir = os.path.join(data_path, output_dir, 'Inference',
                                       'Tile_Scores', 'Epoch_' + str(epoch),
                                       true_test_path)

        if not os.path.isdir(full_output_dir):
            Path(full_output_dir).mkdir(parents=True)

        model_bias_filename = 'bias.xlsx'
        full_model_bias_filename = os.path.join(full_output_dir,
                                                model_bias_filename)
        if not os.path.isfile(full_model_bias_filename):
            try:  # In case this part in not packed in Sequential we'll need this try statement
                last_layer_bias = model.classifier[0].bias.detach().cpu(
                ).numpy()
            except TypeError:
                last_layer_bias = model.classifier.bias.detach().cpu().numpy()

            last_layer_bias_diff = last_layer_bias[1] - last_layer_bias[0]

            last_layer_bias_DF = pd.DataFrame([last_layer_bias_diff])
            last_layer_bias_DF.to_excel(full_model_bias_filename)

        if type(all_slides_tile_scores) == dict:
            all_slides_tile_scores_REG = all_slides_tile_scores['REG']
            all_slides_final_scores_REG = all_slides_final_scores['REG']
            all_slides_tile_scores = all_slides_tile_scores['MIL']
            all_slides_final_scores = all_slides_final_scores['MIL']

            all_slides_tile_scores_REG_DF = pd.DataFrame(
                all_slides_tile_scores_REG[num_model]).transpose()
            all_slides_final_scores_REG_DF = pd.DataFrame(
                all_slides_final_scores_REG[num_model]).transpose()

            tile_scores_file_name_REG = os.path.join(
                data_path, output_dir, 'Inference', 'Tile_Scores',
                'Epoch_' + str(epoch), true_test_path, 'tile_scores_REG.xlsx')
            slide_score_file_name_REG = os.path.join(
                data_path, output_dir, 'Inference', 'Tile_Scores',
                'Epoch_' + str(epoch), true_test_path, 'slide_scores_REG.xlsx')

            all_slides_tile_scores_REG_DF.to_excel(tile_scores_file_name_REG)
            all_slides_final_scores_REG_DF.to_excel(slide_score_file_name_REG)

        all_slides_tile_scores_DF = pd.DataFrame(
            all_slides_tile_scores[num_model]).transpose()
        all_slides_final_scores_DF = pd.DataFrame(
            all_slides_final_scores[num_model]).transpose()
        all_slides_weights_before_sofrmax_DF = pd.DataFrame(
            all_slides_weights_before_softmax[num_model]).transpose()
        all_slides_weights_after_softmax_DF = pd.DataFrame(
            all_slides_weights_after_softmax[num_model]).transpose()

        tile_scores_file_name = os.path.join(data_path, output_dir,
                                             'Inference', 'Tile_Scores',
                                             'Epoch_' + str(epoch),
                                             true_test_path,
                                             'tile_scores.xlsx')
        slide_score_file_name = os.path.join(data_path, output_dir,
                                             'Inference', 'Tile_Scores',
                                             'Epoch_' + str(epoch),
                                             true_test_path,
                                             'slide_scores.xlsx')
        tile_weights_before_softmax_file_name = os.path.join(
            data_path, output_dir, 'Inference', 'Tile_Scores',
            'Epoch_' + str(epoch), true_test_path,
            'tile_weights_before_softmax.xlsx')
        tile_weights_after_softmax_file_name = os.path.join(
            data_path, output_dir, 'Inference', 'Tile_Scores',
            'Epoch_' + str(epoch), true_test_path,
            'tile_weights_after_softmax.xlsx')

        all_slides_tile_scores_DF.to_excel(tile_scores_file_name)
        all_slides_final_scores_DF.to_excel(slide_score_file_name)
        all_slides_weights_before_sofrmax_DF.to_excel(
            tile_weights_before_softmax_file_name)
        all_slides_weights_after_softmax_DF.to_excel(
            tile_weights_after_softmax_file_name)

        logging.info('Tile scores for model {}/{} has been saved !'.format(
            num_model + 1, len(models)))


def extract_tile_scores_for_slide(all_features, models):
    """
    If all_features has shape[0] == 1024, than it;s originated from train type Receptor + is_Tumor.
    In that case we'll need only the first 512 features to compute the tile scores.
    """
    if all_features.shape[0] == 1024:
        all_features = all_features[:512, :]

    # Save tile scores and last models layer bias difference to file:
    tile_scores_list = []
    for index in range(len(models)):
        model = models[index]
        # Compute for each tile the multiplication between its feature vector and the last layer weight difference
        # vector:
        try:  # In case this part in not packed in Sequential we'll need this try statement
            last_layer_weights = model.classifier[0].weight.detach().cpu(
            ).numpy()
        except TypeError:
            last_layer_weights = model.classifier.weight.detach().cpu().numpy()

        f = last_layer_weights[1] - last_layer_weights[0]
        mult = np.matmul(f, all_features.detach().cpu())

        if len(mult.shape) == 1:
            tile_scores_list.append(mult)
        else:
            tile_scores_list.append(mult[:, index])

    return tile_scores_list