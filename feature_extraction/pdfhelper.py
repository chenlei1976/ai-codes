#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import PyPDF2
import pdf2image
import logging
# from pdf2jpg import pdf2jpg
import tempfile


def pdfToText(file_name):
    with open(file_name, 'rb') as f:
        pdfReader = PyPDF2.PdfFileReader(f)
        num_pages = pdfReader.numPages
        print("pages[{}]0".format(num_pages))
        text = ""
        for i in range(num_pages):
            pageObj = pdfReader.getPage(i)
            text += pageObj.extractText()
    return text


def pdfToImage(pdfName, imageName=None):
    try:
        images = pdf2image.convert_from_path(pdfName)
        if len(images) == 0:
            return None
        if imageName is not None:
            images[0].save(imageName, 'PNG')
        return images[0]
    except Exception:
        logging.error('open[%s] failed!' % pdfName)
        return None


# def pdfToJpg(src, dst):
#     try:
#         images = pdf2jpg.convert_pdf2jpg(src, dst, pages="0")
#         print("images[{}]".format(len(images)))
#         return images[0]
#     except Exception:
#         logging.error('open[%s] failed!' % src)
#         return None


if __name__ == '__main__':

    fileName = 'C:\\Users\\Public\\7982599_Law Chung Ming.pdf'
    # fileName = 'C:\\Users\\Public\\delay.pdf'
    saveName = 'C:\\Users\\Public\\test.png'
    pdfToImage(fileName)
    # pdfToJpg(fileName,saveName)
    # if os.path.exists(fileName) and os.path.isfile(fileName):
    #     img = pdfToImage(fileName)
    #     if img is not None:
    #         tmp = np.array(img, dtype=np.uint8)
    #         print(tmp.shape)
    #         img.save(saveName, 'PNG')
