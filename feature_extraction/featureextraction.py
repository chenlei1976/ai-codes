# -*- coding: UTF-8 -*-
import time
import numpy as np
import cv2
import psutil
import pdfhelper
from dbhelper import DBHelper
import filetypecheck
import os
import tools
import logging
import h5py
import datetime
import configparser
try:
    import pypyodbc as pyodbc
except ImportError:
    import pyodbc

k_sql_find_maxId = 'SELECT MAX(Claim_Upload_Id) FROM online.dbo.T_Claim_Upload_Feature'

k_sql_filter_maxId = '''SELECT CU.ID, CU.ClaimID, CU.FilePath, C.PolicyNumber 
FROM online.dbo.T_Claim C, online.dbo.T_Claim_Upload CU 
WHERE CU.ClaimID = C.Id AND CU.ID>{}'''

k_sql_insert = '''INSERT INTO online.dbo.T_Claim_Upload_Feature(Claim_Upload_Id, ClaimID, Feature_File_Path)
VALUES(?,?,?)'''

# k_sql_single_insert = '''INSERT INTO online.dbo.T_Claim_Upload_Feature(Claim_Upload_Id, ClaimID, Feature_File_Path)
# VALUES({},{},{})'''

k_config_file = 'C:\\duplicatecheck\\duplicate_config.ini'
k_log_file = 'C:\\duplicatecheck\\log\\feature-extraction-{}.log'
# k_tmp_img = '.\\tmp.png'


class ExtractionConfig(object):
    def __init__(self):
        files = [k_config_file]
        cfg = configparser.ConfigParser()
        dataset = cfg.read(files)
        if len(dataset) != len(files):
            raise ValueError("Failed to open/find config file")
        # cfg.sections()
        try:
            self.image_size = int(cfg.get('Extraction', 'ImageSize'))
            self.image_base_path = cfg.get('Extraction', 'ImageBasePath')
            self.new_image_path_format = cfg.get('Extraction', 'NewImagePathFormat')
            self.feature_folder = cfg.get('Extraction', 'FeatureFolder')
            self.pdf_to_image_file = cfg.get('Extraction', 'PdfToImageFile')
            self.use_h5 = ('true' == cfg.get('Extraction', 'UseH5'))
            self.include_pdf = ('true' == cfg.get('Extraction', 'IncludePdf'))
            self.insert_batch_size = int(cfg.get('Extraction', 'InsertBatchSize'))
        except configparser.NoSectionError:
            raise ValueError("No section[Extraction] in ini files")


conf = ExtractionConfig()


def createDescriptors(img, detector='surf'):
    if detector.startswith('si'):
        extractor = cv2.xfeatures2d.SIFT_create()
    else:
        extractor = cv2.xfeatures2d.SURF_create()
    _, des = extractor.detectAndCompute(img, None)
    return des


def createFeature(id, fileName, fileType, detector='surf'):
    img = None
    if conf.include_pdf:
        if filetypecheck.isPdfByCustomized(fileType):
            img = pdfhelper.pdfToImage(fileName) # pil image
            if img is None:
                return '', None
            # img.save(k_tmp_img, 'PNG')
            # img = cv2.imread(k_tmp_img)
            img = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif filetypecheck.isImageByCustomized(fileType):
            img = cv2.imread(fileName)
    else:
        if filetypecheck.isImageByCustomized(fileType):
            img = cv2.imread(fileName)

    if img is None:
        if not filetypecheck.isPdfByCustomized(fileType):
            logging.critical('id[{}], not support type[{}], file[{}]!'.format(id, fileType, fileName))
        return '', None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resizeImage(img)

    des = createDescriptors(img, detector=detector)

    path = conf.feature_folder
    if conf.use_h5:
        featureName = os.path.join(path, str(id) + '-' + fileType + '.h5')

        try:
            f = h5py.File(featureName, 'w')
            f.create_dataset('features', data=des)
            f.close()
        except TypeError:
            logging.error('id[{}], h5file[{}] failed!'.format(id, fileName, featureName))
            return '', None
            # raise TypeError('id[{}], h5file[{}] failed!'.format(id, fileName, featureName))

        # Load hdf5 dataset
        # f = h5py.File(featureName, 'r')
        # des = f['features']
        # f.close()
        return featureName, des
    else:
        # use npy file
        featureName = os.path.join(path, str(id) + '-' + fileType + '.npy')
        np.save(featureName, des)
        return featureName, des


def _getRatio(w, h):
    v = float(min(h, w))
    if v < conf.image_size:
        return 1
    return v / conf.image_size


# @tools.timeit
def resizeImage(image):
    h, w = image.shape
    ratio = _getRatio(w, h)
    return cv2.resize(image, (int(w / ratio), int(h / ratio)), interpolation=cv2.INTER_CUBIC)


def handleFiles(id, file1, file2):
    realFile = ''
    fileType = 'unknown'
    tmp = filetypecheck.fileTypeByCustomized(file1)
    if tmp == 'unknown':
        logging.info('id[{}], invalid 1st file[{}]'.format(id, file1))
        tmp = filetypecheck.fileTypeByCustomized(file2)
        if tmp == 'unknown':
            logging.info('id[{}], invalid 2nd file[{}]'.format(id, file2))
        else:
            realFile = file2
            fileType = tmp
    else:
        realFile = file1
        fileType = tmp

    if realFile == '':
        return '', None
    return createFeature(id, realFile, fileType)


def getPossiblePath(policyNumber, claimId, filePath):
    fileName = os.path.basename(filePath)
    # print(','.join(map(str, result)))
    # print(filePath)
    # print(fileName)
    # print("--------")
    subFolder = str(filePath).split("\\\\")[1]
    # print("===== ")
    # print(subFolder)
    # print(conf.image_base_path)
    possibleFile2 = os.path.join(conf.image_base_path, subFolder)
    # print(possibleFile2)
    dateFolder = subFolder.split("\\")[0]

    possibleFile1 = os.path.join(conf.image_base_path, dateFolder,
                    conf.new_image_path_format.format(policyNumber, claimId), fileName)

    return possibleFile1, possibleFile2


def main():
    mainStartTime = time.time()

    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()

    # get max id
    maxId = DBHelper.getValue(conn, k_sql_find_maxId)
    # maxId = 6362
    logging.critical('ID start from {}'.format(maxId))

    # query new files
    cur = conn.cursor()
    cur.execute(k_sql_filter_maxId.format(maxId))

    connInsert = dbhelper.connectDatabase()

    pid = os.getpid()
    proc = psutil.Process(pid)
    params = []
    for result in DBHelper.fetchsome(cur):
        # filePath = result[columns.index('FilePath')]
        fileId = result[0]
        claimId = result[1]
        filePath = result[2]
        policyNumber = result[3]
        possibleFile1, possibleFile2 = getPossiblePath(policyNumber, claimId, filePath)
        featureFile, des = handleFiles(fileId, possibleFile1, possibleFile2)
        if featureFile != '':

            length = len(params)
            if length % 100 == 0:
                # print('params length [{}]'.format(length))
                print("handle id[{}]".format(fileId))
                tools.logMemory(proc)
            # bf = des.tobytes()
            # params.append((fileId, claimId, featureFile, pyodbc.Binary(bf))) # byte value
            #
            params.append((fileId, claimId, featureFile))

            # insert one record
            # DBHelper.insertRecord(connInsert, k_sql_single_insert.format(fileId, claimId, featureFile))

            # optimal performance
            if length >= conf.insert_batch_size:
                logging.info('begin batch insert last file id[{}]'.format(fileId))
                DBHelper.executemany(connInsert, k_sql_insert, params)
                params = []

    length = len(params)
    if (length > 0) and (length < conf.insert_batch_size):
        logging.info('final [%d]batch insert' % length)
        DBHelper.executemany(connInsert, k_sql_insert, params)
    del params
    connInsert.close()
    cur.close()
    conn.close()
    logging.critical('feature extraction take %fs', time.time() - mainStartTime)


if __name__ == "__main__":
    tools.initLog(k_log_file.format(datetime.date.today().strftime('%Y%m%d')))
    main()
