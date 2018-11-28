# -*- coding: UTF-8 -*-
import time
import cv2
from dbhelper import DBHelper
import os
import tools
import logging
import h5py
import datetime
import multiprocessing
from multiprocessing import Pool as ProcessPool
from multiprocessing import freeze_support
from multiprocessing.pool import ThreadPool
import math
import threading

k_use_h5 = True
k_duplicate_threshold = 10
k_near_size = 1000
k_sql_batch_size = 1000
k_log_file = '.\\log\\feature-match-{}.log'

k_sql_unhandled = '''SELECT Claim_Upload_Id,ClaimID,Feature_File_Path 
FROM claim.dbo.T_Claim_Upload_Feature 
WHERE STATUS =0 ORDER BY Claim_Upload_Id'''

k_sql_update_unhandled = '''UPDATE claim.dbo.T_Claim_Upload_Feature 
SET STATUS=1 WHERE Claim_Upload_Id={}'''

k_sql_matched = '''SELECT TOP {} Claim_Upload_Id,Feature_File_Path  
FROM claim.dbo.T_Claim_Upload_Feature 
WHERE Claim_Upload_Id<{} AND ClaimID<>{} ORDER BY Claim_Upload_Id DESC'''

k_sql_duplicate = '''INSERT INTO claim.dbo.T_Claim_Upload_Duplicate(Claim_Upload_Id, Duplicate_Document_Upload_Id, SCORE)  
VALUES(?,?,?)'''

k_desDict = dict()


def getDescriptors(featureFile):
    des = None
    try:
        f = h5py.File(featureFile, 'r')
        des = f['features'][:]
    except Exception:
        logging.error('h5 file read[{}] failed!'.format(featureFile))
    finally:
        if f:
            f.close()
    return des


def featureMatch(des1, des2):
    try:
        bfMatcher = cv2.BFMatcher()
        matches1 = bfMatcher.knnMatch(des1, des2, k=2)
        good1 = [[m] for m, n in matches1 if m.distance < 0.4 * n.distance]

        matches2 = bfMatcher.knnMatch(des2, des1, k=2)
        good2 = [[m] for m, n in matches2 if m.distance < 0.4 * n.distance]
        # logging.critical('similar[{}, {}]'.format(len(good1), len(good2)))
        return min(len(good1), len(good2))
    except ValueError:
        logging.error('match error')
        return 0


def batchMatch(matchingFieldId, matchingFeature, matchedFeatures, threshold):

    print('pid={}, tid={}, fieldId={}, matched size={}'.format(
        os.getpid(), threading.currentThread().ident, matchingFieldId, len(matchedFeatures)))
    duplicateParams = []
    for matchedFieldId, matchedFeature in matchedFeatures:
        num = featureMatch(matchingFeature, matchedFeature)
        if num > threshold:
            duplicateParams.append((matchingFieldId, matchedFieldId, float(num)))
    return duplicateParams


def main():

    tools.initLog(k_log_file.format(datetime.date.today().strftime('%Y%m%d')))

    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()
    cur = conn.cursor()
    # print('exec sql [{}]'.format(k_sql_unhandled))
    cur.execute(k_sql_unhandled)
    connUpdate = dbhelper.connectDatabase()  # update/insert db

    connMatch = dbhelper.connectDatabase()
    cpus = multiprocessing.cpu_count()*2
    for result in DBHelper.fetchsome(cur):
        fileId = int(result[0])
        # print('to fetch [%d], file[%s]' % (fileId, result[2]))
        if fileId in k_desDict.keys():
            desMatching = k_desDict[fileId]
        else:
            desMatching = getDescriptors(result[2])
            if desMatching is not None:
                k_desDict[fileId] = desMatching

        if desMatching is not None:
            claimID = int(result[1])
            curMatch = connMatch.cursor()
            # print('exec sql [{}]'.format(k_sql_matched.format(k_near_size, fileId, claimID)))
            curMatch.execute(k_sql_matched.format(k_near_size, fileId, claimID))

            # collect all matched features
            matchedFeatures = []
            for items in DBHelper.fetchsome(curMatch):
                fileIdMatched = int(items[0])

                if fileIdMatched in k_desDict.keys():
                    desMatched = k_desDict[fileIdMatched]
                else:
                    desMatched = getDescriptors(items[1])
                    if desMatched is not None:
                        k_desDict[fileIdMatched] = desMatched
                if desMatched is not None:
                    matchedFeatures.append((fileIdMatched, desMatched))
            curMatch.close()

            # multiple process to match
            batchSize = math.ceil(len(matchedFeatures)/cpus)
            logging.critical('fileId [{}] to match [{}] features in [{}] process'.format(
                fileId, batchSize, cpus))
            results = []
            beginTime = time.time()
            freeze_support()
            pool = ProcessPool(processes=cpus)
            for batchIter in tools.batch(matchedFeatures, batchSize):
                result = pool.apply_async(batchMatch,
                                          args=(fileId, desMatching, list(batchIter), k_duplicate_threshold))
                results.append(result)
            pool.close()
            pool.join()

            lastResults = []
            for item in results:
                item = item.get()
                if len(item) > 0:
                    lastResults.extend(item)

            if len(results) > 0:
                logging.critical('fileId [{}] has {} similar'.format(fileId, len(lastResults)))
                DBHelper.executemany(connUpdate, k_sql_duplicate, lastResults)

            logging.critical('id[%d] des[%d] match %d records for %fs, dict[%d]' %
                             (fileId, len(desMatching), k_near_size, time.time() - beginTime, len(k_desDict)))
        else:
            logging.error('id[%d] dummy des!', fileId)
        # update status
        DBHelper.updateRecord(connUpdate, k_sql_update_unhandled.format(fileId))

    connUpdate.close()
    connMatch.close()
    cur.close()
    conn.close()


if __name__ == "__main__":
    startTime = time.time()
    main()
    logging.critical('feature extraction take %fs', time.time()-startTime)
