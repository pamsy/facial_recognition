# -*- coding: utf-8 -*-
import logging
import os
import sys
import pickle
import click
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit


FILE_NAME = sys._getframe(  ).f_code.co_filename.split(os.path.sep)[-1]
PROJECT_DIR = str(Path('.').resolve().parent.parent)


def load_data():
    olivetti = fetch_olivetti_faces()
    return olivetti


def bunch2dataframe(B): 
    data = np.c_[B.data, B.target]
    return pd.DataFrame(data)


@click.command()
@click.option('--test_size', type=float, default=1)

def main(test_size, random_state=42):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f'[{FILE_NAME}][{sys._getframe( ).f_code.co_name}][{sys._getframe(  ).f_lineno}] saving raw data')
            
    train_filename = f"olivetti_train_{1-float(test_size)}_raw.data"
    test_filename = f"olivetti_test_{test_size}_raw.data"
    
    RAW_TRAIN_DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'raw', train_filename)
    RAW_TEST_DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'raw', test_filename)        
    
    dataset = load_data()
    dataset_df = bunch2dataframe(dataset)
    

    
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=random_state)
    train_valid_idx, test_idx = next(strat_split.split(dataset.data, dataset.target))
     
    X_train_valid = dataset_df[train_valid_idx]
    X_test = dataset_df[test_idx]
  
    
    try:
        with open(RAW_TRAIN_DATA_FILE, 'wb') as dataset_file:
          pickle.dump(X_train_valid, dataset_file)
    except Exception as e:
        logger.error(f"[{FILE_NAME}][{sys._getframe(  ).f_lineno}] An error occured when saving the raw trainingset. \n Stacktrace : {e}")
        
    
    try:
        with open(RAW_TEST_DATA_FILE, 'wb') as dataset_file:
          pickle.dump(X_test, dataset_file)
    except Exception as e:
        logger.error(f"[{FILE_NAME}][{sys._getframe(  ).f_lineno}] An error occured when saving the raw testingset. \n Stacktrace : {e}")
        

if __name__ == '__main__':

    main()
