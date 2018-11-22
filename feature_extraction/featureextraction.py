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
try:
    import pypyodbc as pyodbc
except ImportError:
    import pyodbc

k_min_image = 300
k_log_file = '.\\log\\feature-detection-{}.log'
k_sift_folder = '.\\siftFeature'
k_surf_folder = '.\\surfFeature'
k_orb_folder = '.\\orbFeature'
k_image_path = 'I:\\05_Claims\\02_eClaim_Files'
k_policyFolder = 'CL{}-{}'
k_use_h5 = True
k_handle_pdf = False
k_sql_find_maxId = 'SELECT MAX(Claim_Upload_Id) FROM claim.dbo.T_Claim_Upload_Feature'

k_sql_filter_maxId = '''SELECT CU.ID, CU.ClaimID, CU.FilePath, C.PolicyNumber 
FROM claim.dbo.T_Claim C, claim.dbo.T_Claim_Upload CU 
WHERE CU.ClaimID = C.Id AND CU.ID>{}'''

k_sql_insert = '''INSERT INTO claim.dbo.T_Claim_Upload_Feature(Claim_Upload_Id, ClaimID, Feature_File_Path, Feature)
VALUES(?,?,?,?)'''

# k_sql_insert = '''INSERT INTO claim.dbo.T_Claim_Upload_Feature(Claim_Upload_Id, ClaimID, Feature_File_Path)
# VALUES(?,?,?)'''

k_sql_single_insert = '''INSERT INTO claim.dbo.T_Claim_Upload_Feature(Claim_Upload_Id, ClaimID, Feature_File_Path, Feature) 
VALUES({},{},{},{})'''

k_sql_insert_batch_size = 100


def createDescriptors(img, detector='surf'):
    if detector.startswith('si'):
        extractor = cv2.xfeatures2d.SIFT_create()
    else:
        extractor = cv2.xfeatures2d.SURF_create()
    _, des = extractor.detectAndCompute(img, None)
    return des


def createFeature(id, fileName, fileType, detector='surf'):
    img = None
    if k_handle_pdf:
        if filetypecheck.isPdfByCustomized(fileType):
            img = pdfhelper.pdfToImage(fileName)
        elif filetypecheck.isImageByCustomized(fileType):
            img = cv2.imread(fileName)
    else:
        if filetypecheck.isImageByCustomized(fileType):
            img = cv2.imread(fileName)

    if img is None:
        if not filetypecheck.isPdfByCustomized(fileType):
            logging.critical('id[{}], not support type[{}], file[{}]!'.format(id, fileType, fileName))
        return '', None

    img = resizeImage(img)

    des = createDescriptors(img, detector=detector)

    if detector.startswith('si'):
        path = k_sift_folder
    else:
        path = k_surf_folder

    if k_use_h5:
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
    if v < k_min_image:
        return 1
    return v / k_min_image


# @tools.timeit
def resizeImage(image):
    h, w, c = image.shape
    ratio = _getRatio(w, h)
    return cv2.resize(image, (int(w / ratio), int(h / ratio)), interpolation=cv2.INTER_CUBIC)


def handleFiles(id, file1, file2):
    realFile = ''
    fileType = 'unknown'
    tmp = filetypecheck.fileTypeByCustomized(file1)
    if tmp == 'unknown':
        logging.warning('id[{}], invalid 1st file[{}]'.format(id, file1))
        tmp = filetypecheck.fileTypeByCustomized(file2)
        if tmp == 'unknown':
            logging.warning('id[{}], invalid 2nd file[{}]'.format(id, file2))
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

    subFolder = str(filePath).split("\\\\")[1]
    possibleFile2 = os.path.join(k_image_path, subFolder)

    dateFolder = subFolder.split("\\")[0]

    possibleFile1 = os.path.join(k_image_path, dateFolder, k_policyFolder.format(policyNumber, claimId), fileName)

    return possibleFile1, possibleFile2


def main():

    tools.initLog(k_log_file.format(datetime.date.today().strftime('%Y%m%d')))
    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()

    # get max id
    maxId = DBHelper.getValue(conn, k_sql_find_maxId)
    # maxId = 27736
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
            # print(fileId, featureFile)

            bf = des.tobytes()
            length = len(params)
            if length % 50 == 0:
                print('params length [{}]'.format(length))
                tools.logMemory(proc)
            params.append((fileId, claimId, featureFile, pyodbc.Binary(bf)))
            # params.append((fileId, claimId, featureFile))

            # insert one record
            # DBHelper.insertRecord(connInsert, k_sql_single_insert.format(result[0], result[1], featureFile, des.tobytes()))

            # optimal performance
            if length >= k_sql_insert_batch_size:
                logging.info('begin batch insert')
                DBHelper.executemany(connInsert, k_sql_insert, params)
                params = []

    length = len(params)
    if (length > 0) and (length < k_sql_insert_batch_size):
        logging.info('final [%d]batch insert' % length)
        DBHelper.executemany(connInsert, k_sql_insert, params)
    del params
    connInsert.close()
    cur.close()
    conn.close()


if __name__ == "__main__":
    startTime = time.time()
    main()
    logging.critical('feature extraction take %fs', time.time()-startTime)