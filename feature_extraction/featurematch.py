# -*- coding: UTF-8 -*-
import time
import cv2
import psutil
from dbhelper import DBHelper
import os
import tools
import logging
import h5py
import datetime

k_use_h5 = True
k_duplicate_threshold = 10
k_near_size = 1000
k_sql_batch_size = 1000
k_log_file = '.\\log\\feature-match-{}.log'

k_sql_unhandled = '''SELECT Claim_Upload_Id,ClaimID,Feature_File_Path 
FROM claim.dbo.T_Claim_Upload_Feature 
WHERE Claim_Upload_Id IN (27960,27961) AND STATUS =0 ORDER BY Claim_Upload_Id'''

k_sql_update_unhandled = '''UPDATE claim.dbo.T_Claim_Upload_Feature 
SET STATUS=1 WHERE Claim_Upload_Id={}'''

k_sql_matched = '''SELECT TOP {} Claim_Upload_Id,Feature_File_Path  
FROM claim.dbo.T_Claim_Upload_Feature 
WHERE Claim_Upload_Id<{} AND ClaimID<>{} ORDER BY Claim_Upload_Id DESC'''

k_sql_duplicate = '''INSERT INTO claim.dbo.T_Claim_Upload_Duplicate(Claim_Upload_Id, Duplicate_Document_Upload_Id, SCORE)  
VALUES(?,?,?)'''

k_bfMatcher = cv2.BFMatcher()
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


def loadH5(fileName):
    f = h5py.File(fileName, 'r')
    des = f['features'][:]
    return des


def featureMatch(des1, des2):
    # BFMatcher with default params

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # bf = cv2.FlannBasedMatcher(index_params, search_params)

    # bf = cv2.BFMatcher()

    try:
        matches1 = k_bfMatcher.knnMatch(des1, des2, k=2)
        good1 = [[m] for m, n in matches1 if m.distance < 0.4 * n.distance]

        matches2 = k_bfMatcher.knnMatch(des2, des1, k=2)
        good2 = [[m] for m, n in matches2 if m.distance < 0.4 * n.distance]

        # logging.critical('similar[{}, {}]'.format(len(good1), len(good2)))
        return min(len(good1), len(good2))
    except ValueError:
        logging.error('match error')
        return 0


def main():

    tools.initLog(k_log_file.format(datetime.date.today().strftime('%Y%m%d')))

    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()
    cur = conn.cursor()

    cur.execute(k_sql_unhandled)

    connUpdate = dbhelper.connectDatabase()  # update/insert db

    connMatch = dbhelper.connectDatabase()

    for result in DBHelper.fetchsome(cur):
        fileId = int(result[0])
        if fileId in k_desDict.keys():
            desMatching = k_desDict[fileId]
        else:
            desMatching = getDescriptors(result[2])
            k_desDict[fileId] = desMatching

        if desMatching is not None:
            claimID = int(result[1])
            curMatch = connMatch.cursor()
            # print('exec sql [{}]'.format(k_sql_matched.format(k_near_size, fileId, claimID)))
            curMatch.execute(k_sql_matched.format(k_near_size, fileId, claimID))
            duplicateParams = []  # insert duplicate features
            beginTime = time.time()
            for records in DBHelper.fetchsome(curMatch):
                fileIdMatched = int(records[0])

                if fileIdMatched in k_desDict.keys():
                    desMatched = k_desDict[fileIdMatched]
                else:
                    desMatched = getDescriptors(records[1])
                    k_desDict[fileIdMatched] = desMatched

                if desMatched is not None:
                    num = featureMatch(desMatching, desMatched)
                    if num > k_duplicate_threshold:
                        logging.critical('{}>>{} matching {}>>{}, similar[{}]'.format(
                            fileId, result[2], fileIdMatched, records[1], num))
                        duplicateParams.append((fileId, fileIdMatched, float(num)))
                        if len(duplicateParams) >= k_sql_batch_size:
                            logging.info('begin batch insert duplicate features')
                            DBHelper.executemany(connUpdate, k_sql_duplicate, duplicateParams)
                            duplicateParams = []
                        # DBHelper.insertRecord(curMatch, k_sql_duplicate.format(id, records[0], float(num)))
            length = len(duplicateParams)
            if (length > 0) and (length < k_sql_batch_size):
                logging.info('final [%d]batch insert duplicate features' % length)
                DBHelper.executemany(connUpdate, k_sql_duplicate, duplicateParams)
            del duplicateParams

            logging.critical('id[%d] des[%d] match %d records for %fs, dict[%d]' %
                             (fileId, len(desMatching), k_near_size, time.time() - beginTime, len(k_desDict)))
            curMatch.close()
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

    # des1 =loadH5('.\\surfFeature\\6247-jpg.h5')
    #
    # des2 =loadH5('.\\surfFeature\\6316-jpg.h5')
    #
    # num = featureMatch(des1, des2)
    # print('similar[{}]'.format(num))
    #
    # num = featureMatch(des2, des1)
    # print('similar[{}]'.format(num))

