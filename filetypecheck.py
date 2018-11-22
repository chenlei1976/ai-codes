#!/usr/bin/python
# -*- coding: UTF-8 -*-

import filetype
import struct
import os

k_fileTypeDict = {
    'FFD8FF': 'jpg',
    '89504E47': 'png',
    '47494638': 'gif',
    '49492A00': 'tif',
    '424D': 'bmp',
    '41433130': 'dwg',
    '38425053': 'psd',
    '7B5C727466': 'rtf',
    '3C3F786D6C': 'xml',
    '68746D6C3E': 'html',
    '44656C69766572792D646174653A': 'eml',
    'CFAD12FEC5FD746F': 'dbx',
    '2142444E': 'pst',
    'D0CF11E0': 'doc/xls',
    '5374616E64617264204A': 'mdb',
    'FF575043': 'wpd',
    '252150532D41646F6265': 'ps/eps',
    '255044462D312E': 'pdf',
    'AC9EBD8F': 'qdf',
    'E3828596': 'pwl',
    '504B0304': 'zip',
    '52617221': 'rar',
    '57415645': 'wav',
    '41564920': 'avi',
    '2E7261FD': 'ram',
    '2E524D46': 'rm',
    '000001BA': 'mpg',
    '000001B3': 'mpg',
    '6D6F6F76': 'mov',
    '3026B2758E66CF11': 'asf',
    '4D546864': 'mid',
    "4D5A900003": "exe",
    "D0CF11E0": "docx"
}


def bytes2hex(bytes):
    num = len(bytes)
    hexStr = u""
    for i in range(num):
        t = u"%x" % bytes[i]
        if len(t) % 2:
            hexStr += u"0"
        hexStr += t
    return hexStr.upper()


def isPdfByCustomized(fileType):
    return fileType == 'pdf'


def isImageByCustomized(fileType):
    if fileType in ('jpg', 'png', 'gif', 'tif', 'bmp'):
        return True
    return False


def fileTypeByCustomized(fileName):
    fType = 'unknown'
    if os.path.exists(fileName) and os.path.isfile(fileName):
        if fileName.lower().endswith('txt'):
            return 'txt'
        with open(fileName, 'rb') as f:
            for k, v in k_fileTypeDict.items():
                numOfBytes = int(len(k) / 2)
                f.seek(0)
                try:
                    hBytes = struct.unpack('B' * numOfBytes, f.read(numOfBytes))
                except struct.error:
                    return fType
                tmpCode = bytes2hex(hBytes)
                if tmpCode == k:
                    fType = v
                    break
    # print("file [{}], type[{}]".format(fileName, fType))
    return fType


def getFileType(fileName):
    try:
        kind = filetype.guess(fileName)
        if kind is None:
            print('Cannot guess file type [{}]!'.format(fileName))
            return None, None
        else:
            print('File name[{}], extension[{}], MIME[{}]'.format(fileName, kind.extension, kind.mime))
            return kind.extension, kind.mime
    except FileNotFoundError:
        print("No file [{}]".format(fileName))
        return None, None


def isPdf(extension):
    return extension == 'pdf'


def isImage(mime):
    return mime.startswith('image')


def main():
    files = [
        '/home/chenlei/06.mp4',
        '/home/chenlei/ovt.pdf',
        '/home/chenlei/clinic1.png',
        '/home/chenlei/chenlei-ic.jpg',
        '/home/chenlei/ta.PNG',
        '/home/chenlei/6817-luggage 1.JPG',
        '/home/chenlei/7553-luggage.jpeg',
        '/home/chenlei/ptb.train.txt',
        '/home/chenlei/11.txt',
            ]

    for fileName in files:
        # getFileType(fileName)
        fileTypeByCustomized(fileName)


if __name__ == '__main__':
    main()
    
