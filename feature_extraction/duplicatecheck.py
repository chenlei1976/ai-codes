#!/usr/bin/python
# -*- coding: UTF-8 -*-
import datetime
import os

import tools
import featureextraction
import featurematchformultipleprocess
import time
import logging

k_log_file = 'C:\\duplicatecheck\\log\\duplicate-check-{}.log'
k_run_file = 'C:\\duplicatecheck\\duplicate-running.txt'

if __name__ == '__main__':
    # tools.initLog(k_log_file.format(datetime.date.today().strftime('%Y%m%d')))
    if os.path.exists(k_run_file):
        print('last duplicate is still running')
    else:
        tools.initLog(k_log_file.format(datetime.datetime.now().strftime('%Y%m%d%H%M')))
        try:
            logging.critical('create running file')
            file = open(k_run_file, 'w')
            file.close()

            startTime1 = time.time()
            featureextraction.main()
            startTime2 = time.time()
            logging.critical('feature extraction take %fs' % (startTime2 - startTime1))
            featurematchformultipleprocess.main()
            logging.critical('feature match take %fs' % (time.time() - startTime2))

            logging.critical('remove running file')
            os.remove(k_run_file)
        except Exception:
            logging.error('critical bug')
            os.remove(k_run_file)
