# -*- coding: UTF-8 -*-
import configparser
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


k_log_file = 'C:\\duplicatecheck\\log\\feature-match-{}.log'

k_sql_unhandled = '''SELECT TOP {} Claim_Upload_Id,ClaimID,Feature_File_Path 
FROM online.dbo.T_Claim_Upload_Feature 
WHERE STATUS =0 ORDER BY Claim_Upload_Id'''

# Claim_Upload_Id IN (25142,25143) AND
# Claim_Upload_Id IN (27960,27961) AND
# Claim_Upload_Id IN (27950,27921) AND

k_sql_update_unhandled = '''UPDATE online.dbo.T_Claim_Upload_Feature 
SET STATUS=1 WHERE Claim_Upload_Id={}'''

k_sql_matched = '''SELECT TOP {} Claim_Upload_Id,Feature_File_Path  
FROM online.dbo.T_Claim_Upload_Feature 
WHERE Claim_Upload_Id<{} AND ClaimID<>{} ORDER BY Claim_Upload_Id DESC'''

k_sql_duplicate = '''INSERT INTO online.dbo.T_Claim_Upload_Duplicate(Claim_Upload_Id, Duplicate_Document_Upload_Id, Score, Method, Status)  
VALUES(?,?,?,?,?)'''


class MatchConfig(object):
    def __init__(self):
        files = ['C:\\duplicatecheck\\duplicate_config.ini']
        cfg = configparser.ConfigParser()
        dataset = cfg.read(files)
        if len(dataset) != len(files):
            raise ValueError("Failed to open/find config file")
        # cfg.sections()
        try:
            self.duplicate_threshold = int(cfg.get('Match', 'DuplicateThreshold'))
            self.feature_different_ratio = float(cfg.get('Match', 'FeatureDifferentRatio'))
            self.search_scope = int(cfg.get('Match', 'SearchScope'))
            self.max_check_images = int(cfg.get('Match', 'MaxCheckImages'))
            self.max_similarity = int(cfg.get('Match', 'MaxSimilarity'))
            self.similarity_distance = float(cfg.get('Match', 'SimilarityDistance'))
        except configparser.NoSectionError:
            raise ValueError("No section[Match] in ini files")


k_desDict = dict()
conf = MatchConfig()
k_diff_ratio = conf.feature_different_ratio
k_similarity_distance = conf.similarity_distance

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
        lenDes1 = len(des1)
        lenDes2 = len(des2)
        if lenDes1 == 0 or lenDes2 == 0:
            return 0
        minLen = min(lenDes1, lenDes2)
        maxLen = max(lenDes1, lenDes2)
        if (maxLen / minLen) > k_diff_ratio:
            return 0

        bfMatcher = cv2.BFMatcher()
        matches1 = bfMatcher.knnMatch(des1, des2, k=2)
        good1 = [[m] for m, n in matches1 if m.distance < k_similarity_distance * n.distance]

        matches2 = bfMatcher.knnMatch(des2, des1, k=2)
        good2 = [[m] for m, n in matches2 if m.distance < k_similarity_distance * n.distance]
        # logging.critical('similar[{}, {}]'.format(len(good1), len(good2)))
        return min(len(good1), len(good2))
    except ValueError:
        logging.error('match error')
        return 0


def batchMatch(matchingFieldId, matchingFeature, matchedFeatures, threshold):
    # print('pid={}, tid={}, fieldId={}, matched size={}'.format(
    #     os.getpid(), threading.currentThread().ident, matchingFieldId, len(matchedFeatures)))
    duplicateParams = []
    for matchedFieldId, matchedFeature in matchedFeatures:
        num = featureMatch(matchingFeature, matchedFeature)
        if num > threshold:
            duplicateParams.append((matchingFieldId, matchedFieldId, float(num), 0, 0))
    return duplicateParams


def main():
    # logging.info('main')
    mainStartTime = time.time()
    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()
    cur = conn.cursor()
    # print('exec sql [{}]'.format(k_sql_unhandled))
    # logging.info('exec sql')
    cur.execute(k_sql_unhandled.format(conf.max_check_images))
    # logging.info('exec new connect1')
    connUpdate = dbhelper.connectDatabase()  # update/insert db
    # logging.info('exec new connect2')
    connMatch = dbhelper.connectDatabase()
    cpus = multiprocessing.cpu_count() * 2
    # logging.info('exec fetchsome, cpu[{}]'.format(cpus))
    for result in DBHelper.fetchsome(cur):
        fileId = int(result[0])
        # print('to fetch [%d], file[%s]' % (fileId, result[2]))
        if fileId in k_desDict.keys():
            matchingFeature = k_desDict[fileId]
        else:
            matchingFeature = getDescriptors(result[2])
            if matchingFeature is not None:
                k_desDict[fileId] = matchingFeature

        if matchingFeature is not None:
            # logging.info('matchingFeature')
            claimID = int(result[1])
            curMatch = connMatch.cursor()
            # print('to fetch [%d], [%d]features, file[%s]' % (fileId, len(matchingFeature), result[2]))
            print('exec sql [{}]'.format(k_sql_matched.format(conf.search_scope, fileId, claimID)))
            curMatch.execute(k_sql_matched.format(conf.search_scope, fileId, claimID))

            # collect all matched features
            matchedFeatures = []
            # startFetchSome = time.time()
            # print('fetch some begin')
            # logging.info('fetch some begin')
            for items in DBHelper.fetchsome(curMatch):
                fileIdMatched = int(items[0])

                if fileIdMatched in k_desDict.keys():
                    matchedFeature = k_desDict[fileIdMatched]
                else:
                    matchedFeature = getDescriptors(items[1])
                    if matchedFeature is not None:
                        k_desDict[fileIdMatched] = matchedFeature
                if matchedFeature is not None:
                    matchedFeatures.append((fileIdMatched, matchedFeature))
            curMatch.close()
            # print('fetch some end [%fs]',time.time() - startFetchSome)
            # logging.info('fetch some end')
            # print('fetch some end')

            # multiple process to match
            batchSize = math.ceil(len(matchedFeatures)/cpus)
            logging.critical('fileId [{}]=[{}]features to match [{}] records in [{}] process'.format(
                fileId, len(matchingFeature), batchSize, cpus))
            results = []
            beginTime = time.time()
            freeze_support()
            pool = ProcessPool(processes=cpus)
            for batchIter in tools.batch(matchedFeatures, batchSize):
                result = pool.apply_async(batchMatch,
                                          args=(fileId, matchingFeature, list(batchIter), conf.duplicate_threshold))
                results.append(result)
            pool.close()
            pool.join()

            lastResults = []
            for item in results:
                item = item.get()
                if len(item) > 0:
                    lastResults.extend(item)

            similarNum = len(lastResults)
            if similarNum > 0:
                logging.critical('fileId[{}] choose {} in {} duplications'.format(fileId, conf.max_similarity, similarNum))
                lastResults = sorted(lastResults, key=lambda x: x[2], reverse=True)[:conf.max_similarity]
                DBHelper.executemany(connUpdate, k_sql_duplicate, lastResults)
            else:
                logging.critical('fileId[{}] no similar image'.format(fileId))
            logging.critical('id[%d] des[%d] match %d records for %fs, dict[%d]' %
                             (fileId, len(matchingFeature), conf.search_scope, time.time() - beginTime, len(k_desDict)))
        else:
            logging.error('id[%d] dummy des!', fileId)
        # update status
        DBHelper.updateRecord(connUpdate, k_sql_update_unhandled.format(fileId))

    connUpdate.close()
    connMatch.close()
    cur.close()
    conn.close()
    logging.critical('feature matching take %fs', time.time() - mainStartTime)


if __name__ == "__main__":
    cpus = multiprocessing.cpu_count()
    print("number of CPU is %d" % cpus)
    tools.initLog(k_log_file.format(datetime.date.today().strftime('%Y%m%d')))
    main()


