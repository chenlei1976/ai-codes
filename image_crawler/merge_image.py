#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import os
import shutil
import tools
from PIL import Image
import psutil
import pytesseract


def _merge_image(src_folder, dst_folder, prefix_name):
    pid = os.getpid()
    p = psutil.Process(pid)
    logging.info('Process id[{}] name[{}]'.format(pid, p.name()))
    src_images = tools.get_image_files(src_folder)
    logging.info('{} files in "{}"'.format(len(src_images), src_folder))
    for index, src_image in enumerate(src_images):

        if index % 10 == 0:
            info = p.memory_full_info()
            memory = info.uss / 1024. / 1024.
            logging.info('Memory used: {:.2f} MB'.format(memory))

        dst_images = tools.get_image_files(dst_folder, False)
        if not tools.has_same_image(src_image, dst_images):
            new_name = dst_folder + os.sep + prefix_name + '_' + str(len(dst_images) + 1) + '.png'
            img = Image.open(src_image)
            img.convert('RGB').save(new_name, "PNG")  # this converts image as jpeg
            os.remove(src_image)
            # logging.warning('to move "{}" -> "{}"'.format(src_image, new_name)))
            # shutil.move(src_image, new_name)
        else:
            logging.warning('same image [{}]'.format(src_image))


if __name__ == '__main__':

    tools.init_log('./merge_image.log')

    # testing
    # dst = '/home/chenlei/images/etiqa/boarding_pass'
    # src = '/home/chenlei/images/google/test'
    #
    # if not os.path.exists(dst):
    #     logging.critical('create folder [{}]'.format(dst))
    #     os.mkdir(dst)
    # _merge_image(src, dst, "boarding_pass")


    # real
    src_boarding_pass = ['/home/chenlei/images/google/boarding pass',
                         '/home/chenlei/images/bing/boarding pass',
                         '/home/chenlei/images/yahoo/boarding pass']

    src_luggage = ['/home/chenlei/images/google/luggage',
                   '/home/chenlei/images/bing/luggage',
                   '/home/chenlei/images/yahoo/luggage']

    src_receipt = ['/home/chenlei/images/google/receipt',
                   '/home/chenlei/images/bing/receipt',
                   '/home/chenlei/images/yahoo/receipt']

    src_clinic_receipt = ['/home/chenlei/images/google/clinic receipt',
                          '/home/chenlei/images/bing/clinic receipt',
                          '/home/chenlei/images/yahoo/clinic receipt']

    src_hospital_receipt = ['/home/chenlei/images/google/hospital receipt',
                            '/home/chenlei/images/bing/hospital receipt',
                            '/home/chenlei/images/yahoo/hospital receipt']

    src_passport = ['/home/chenlei/images/google/passport first pages',
                    '/home/chenlei/images/bing/passport first pages',
                    '/home/chenlei/images/yahoo/passport first pages']

    src_broken_pad = ['/home/chenlei/images/google/broken ipad',
                      '/home/chenlei/images/bing/broken ipad',
                      '/home/chenlei/images/yahoo/broken ipad']

    src_broken_phone = ['/home/chenlei/images/google/broken mobile phone',
                        '/home/chenlei/images/bing/broken mobile phone',
                        '/home/chenlei/images/yahoo/broken mobile phone']

    dst_path = '/home/chenlei/images/etiqa/'

    dicts = {'boarding_pass': src_boarding_pass,
             'luggage': src_luggage,
             'receipt': src_receipt,
             'clinic_receipt': src_clinic_receipt,
             'hospital_receipt': src_hospital_receipt,
             'passport': src_passport,
             'broken_pad': src_broken_pad,
             'broken_phone': src_broken_phone
             }

    for key, value in dicts.items():
        dst_folder = dst_path + key
        if not os.path.exists(dst_folder):
            logging.critical('create folder [{}]'.format(dst_folder))
            os.mkdir(dst_folder)

        for src_folder in value:
            if os.path.exists(src_folder):
                _merge_image(src_folder, dst_folder, key)
            else:
                logging.error('not exist folder [{}]'.format(src_folder))
