# -*- coding: UTF-8 -*-
import argparse
import logging
import os
import shutil
from PIL import Image
import numpy as np
import dHash
import time


def init_log(logfile):
    # step1, create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # step2, create a handler for file
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    # step3, create a handler for onsole
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # step4， define formatter
    LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    DATE_FORMAT = "%d/%m/%Y %H:%M:%S %p"
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # step5，add logger into handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    # logging.debug("This is a debug log.")
    # logging.info("This is a info log.")
    # logging.warning("This is a warning log.")
    # logging.error("This is a error log.")
    # logging.critical("This is a critical log.")


def timeit(func):
    """
    Prints time taken for a function call
    :param func: function call
    :return: Computes the time taken for each function call
    """

    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        logging.info('function [%s] take %fs' % (func.__name__, elapsedTime))

    return wrapper


def verify_image(img_file):
    try:
        # logging.debug('to open "{}"'.format(img_file))
        v_img = Image.open(img_file)
        v_img.verify()
        return True
    except OSError:
        logging.error('OSError "{}"'.format(img_file))
        return False
    except IOError:
        logging.error('IOError "{}"'.format(img_file))
        return False


def get_files(file_path):
    files = [file_path + os.sep + f for f in os.listdir(file_path) if os.path.isfile(file_path + os.sep + f)]
    return files


def get_image_files(img_path, need_check=True):
    files = get_files(img_path)
    # logging.debug('{} files in "{}"'.format(len(files), img_path))
    if not need_check:
        return files

    image_files = []
    for img_file in files:
        if verify_image(img_file):
            image_files.append(img_file)
        else:
            logging.error('to delete invalid image file [{}]'.format(img_file))
            os.remove(img_file)
            # shutil.move(img_file, '/tmp/')
    return image_files


def is_same(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())


def has_same_image(image_file, image_list):
    img = Image.open(image_file)
    for tmp in image_list:
        if is_same(img, Image.open(tmp)):
            return True
    return False


def stop_words():
    """
    keep 'from' delete 'to'
    """
    return ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
            "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "to"]


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", required=True,
                    help="path to input dataset_test")
    # ap.add_argument("-dtrain", "--dataset_train", required=True,
    #                 help="path to input dataset_train")
    # ap.add_argument("-m", "--model", required=True,
    #                 help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    pass
