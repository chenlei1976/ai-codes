#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import PyPDF2
import pdf2image
from filetypecheck import fileTypeByCustomized
from pdfminer.pdfparser import  PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed


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


def parse(pdfFile, output=None):
    with open(pdfFile, 'rb') as fp:
        parser = PDFParser(fp)
        doc = PDFDocument()
        parser.set_document(doc)
        doc.set_parser(parser)

        doc.initialize()
        if not doc.is_extractable:
            print('pdf is not extractable')
            return None
        else:
            rsrcmgr = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(rsrcmgr, laparams=laparams)
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in doc.get_pages():
                interpreter.process_page(page)
                layout = device.get_result()
                for x in layout:
                    if isinstance(x, LTTextBoxHorizontal):
                        results = x.get_text()
                        if output is not None:
                            with open(output, 'a') as f:
                                results = x.get_text()
                                # print(results)
                                f.write(results + os.linesep)

if __name__ == '__main__':

    folder = 'C:\\pdf'
    # print(os.listdir(folder))
    # outFile = ' '
    # parse(fileName, output=outFile)

    for file_name in os.listdir(folder):
        full_file = os.path.join(folder, file_name)
        if os.path.isfile(full_file) and 'pdf' == fileTypeByCustomized(full_file):
            output_file = os.path.splitext(full_file)[0]+'.txt'
            parse(full_file, output=output_file)

