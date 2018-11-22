#!/usr/bin/python
# -*- coding: UTF-8 -*-
import PyPDF2
import pdf2image
import logging

def pdfToText(file_name):

    with open(file_name, 'rb') as f:
        pdfReader = PyPDF2.PdfFileReader(f)
        num_pages = pdfReader.numPages
        text = ""
        for i in range(num_pages):
            pageObj = pdfReader.getPage(i)
            text += pageObj.extractText()

    return text


def pdfToImage(pdfName, imageName=None):
    try:
        images = pdf2image.convert_from_path(pdfName)
        print("len[{}] file[{}]]".format(len(images), pdfName))
        if len(images) == 0:
            return None
        if imageName is not None:
            images[0].save(imageName, 'PNG')
        return images[0]
    except Exception:
        logging.error('open[%s] failed!' % pdfName)
        return None

if __name__ == '__main__':
    img = pdfToImage('C:\\Users\\Public\\Boarding Passes.pdf')
