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
k_log_file = 'C:\\Users\\Public\\duplicatecheck\\log\\dbtest-{}.log'
k_surf_folder = 'C:\\Users\\Public\\duplicatecheck\\surfFeature'
k_sql_select = '''SELECT Claim_Upload_Id,Feature_File_Path 
FROM claim.dbo.T_Claim_Upload_Feature'''

k_sql_update = '''UPDATE claim.dbo.T_Claim_Upload_Feature 
SET Feature_File_Path='{}' WHERE Claim_Upload_Id={}'''

def main():

    dbhelper = DBHelper()
    conn = dbhelper.connectDatabase()
    # query new files
    cur = conn.cursor()
    cur.execute(k_sql_select)

    params = []
    for result in DBHelper.fetchsome(cur):
        fileId = result[0]
        oldFile = result[1]
        if not oldFile.startswith('C'):
            newFile = os.path.join(k_surf_folder, os.path.basename(oldFile))
            # print('id[{}] == file[{}]'.format(fileId, newFile))
            params.append((fileId, newFile))

    cur.close()
    print('size[{}]' .format(len(params)))
    for item in params:
        print('update id[{}]'.format(item[0]))
        DBHelper.updateRecord(conn, k_sql_update.format(item[1], item[0]))
        time.sleep(0.3)
    conn.close()


if __name__ == "__main__":
    tools.initLog(k_log_file.format(datetime.date.today().strftime('%Y%m%d')))
    main()
