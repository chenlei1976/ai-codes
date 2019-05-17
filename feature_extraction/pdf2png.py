#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import pdfhelper
from filetypecheck import fileTypeByCustomized
from PIL import Image
from PIL import ImageFile
import tools
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True

k_pdf_path = 'C:\\Users\\Public\\workspace\\classifier_model\\datasets\\pdf'
k_png_path = 'C:\\Users\\Public\\workspace\\classifier_model\\datasets\\png'

if __name__ == '__main__':
    tools.initLog('C:\\Users\\Public\\workspace\\classifier_model\\datasets\\png\\pdf2png.log')
    for file_name in os.listdir(k_pdf_path):
        full_path = os.path.join(k_pdf_path, file_name)
        if os.path.isfile(full_path) and 'pdf' == fileTypeByCustomized(full_path):
            img = pdfhelper.pdfToImage(full_path) # pil image
            if img is not None:
                output_file = os.path.join(k_png_path,  os.path.splitext(file_name)[0]+'.png')
                print(output_file)
                if not os.path.exists(output_file):
                    try:
                        img.save(output_file)
                    except Exception:
                        logging.error('save[%s] failed!' % output_file)