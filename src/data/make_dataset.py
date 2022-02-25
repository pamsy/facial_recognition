# -*- coding: utf-8 -*-
import click
import logging
import sys
import os
import pickle
import _pickle as cPickle

from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from sklearn.decomposition import PCA


FILE_NAME = sys._getframe(  ).f_code.co_filename.split(os.path.sep)[-1]

logger = logging.getLogger(__name__)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info(f'[{FILE_NAME}][{sys._getframe( ).f_code.co_name}][{sys._getframe(  ).f_lineno}] making final data set from raw data')
    
    X = load(input_filepath)
    X_pca = compress_images(X)
    filename = getFileName(input_filepath)
    save(X_pca,'processed_'+ filename)


    
    
def compress_images(X, variance_rate=0.99):
    logger.info(f'[{FILE_NAME}][{sys._getframe( ).f_code.co_name}][{sys._getframe(  ).f_lineno}] start compress_images')

    
    pca = PCA(variance_rate)
    X_pca = pca.fit_transform(X)
    
    return X_pca

def load(filename):
    logger.info(f'[{FILE_NAME}][{sys._getframe( ).f_code.co_name}][{sys._getframe(  ).f_lineno}] load file')

    try:
        with open(filename, "rb") as input_file:
            e = cPickle.load(input_file)
            return e
    except Exception as e:
        logger.error(f'[{FILE_NAME}][{sys._getframe( ).f_code.co_name}][{sys._getframe(  ).f_lineno}] an error occured when loading the file. \n Stacktrace : {e}')
    
    
def getFileName(f):
    logger.info(f'[{FILE_NAME}][{sys._getframe( ).f_code.co_name}][{sys._getframe(  ).f_lineno}] get file name from path')

    filename = f.split(os.path.sep)[-1]
    return filename

def save(X, filename):
    logger.info(f'[{FILE_NAME}][{sys._getframe( ).f_code.co_name}][{sys._getframe(  ).f_lineno}] save an object into a pickle file')

    PROCESSED_DATA_FILE = os.path.join(str(Path('.').resolve().parent.parent), 'data', 'processed', filename) 
    try:
        with open(PROCESSED_DATA_FILE, 'wb') as dataset_file:
          pickle.dump(X, dataset_file)
    except Exception as e:
        logger.error(f"[{FILE_NAME}][{sys._getframe(  ).f_lineno}] An error occured when saving the raw trainingset. \n Stacktrace : {e}")
        

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
        
    main()
