#!/usr/bin/python
# -*- coding: UTF-8 -*-
import PyPDF2
import pdf2image


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
    images = pdf2image.convert_from_path(pdfName)
    if len(images) == 0:
        pass
    if imageName is not None:
        images[0].save(imageName, 'PNG')
    return images[0]


if __name__ == '__main__':
    img = pdfToImage('./ta.pdf')
