#!/usr/bin/python
# -*- coding: UTF-8 -*-

import logging
import os
import re
import shutil
import tools
from PIL import Image
import psutil
import pytesseract
from nltk.corpus import brown
import enchant

# import nltk
# from nltk.corpus import stopwords

k_min_word_limit = 2
k_suggest_words = 2
word_dict = enchant.Dict("en_US")


def _test1():
    file_name = '/home/chenlei/images/boarding_pass_15.png'
    image = Image.open(file_name)
    code = pytesseract.image_to_string(image)  # return unicode
    print(code)

    tokens = _process_text(code)
    print(tokens)

    # tokens = [u'carrier', u'boarding', u'pass', u'scanner', u'simply', u'click', u'the', u'then', u'point', u'your',
    #           u'camera', u'samari', u'llc']
    # # stop_words = set(stopwords.words('english'))
    # stop_words = tools.stop_words()
    # tokens = [w for w in tokens if w not in stop_words]
    # print tokens[:]
    # print ' '.join(tokens)

    # word_list = brown.words()
    # word_set = set(word_list)

    # word_dict = enchant.Dict("en_US")
    # print word_dict.check("hello")
    # print word_dict.check("helo")
    # print word_dict.suggest("llc")
    # print word_dict.suggest("samari")


def _filter_words(tokens):
    if not isinstance(tokens, (list, tuple, set)):
        logging.error("input must be a list/tuple/set")
        raise TypeError("input must be a list/tuple/set")
    stop_words = tools.stop_words()
    # tokens = [t.lower() for t in tokens if len(t) > k_min_word_limit and re.match("^[A-Za-z]*$", t)]
    tokens = [t.lower() for t in tokens if re.match("^[A-Za-z]*$", t)]

    words = []
    for w in tokens:
        if len(w) == 2 and word_dict.check(w):
            words.append(w)
        if len(w) == 3 and word_dict.check(w):
            words.append(w)
        if len(w) > 3:
            words.append(w)

    return [w for w in words if w not in stop_words]


def _process_text(raw_text):
    tokens = _filter_words(raw_text.split())
    spell_list = []
    for w in tokens:
        if not word_dict.check(w):
            suggests = _filter_words(word_dict.suggest(w)[:k_suggest_words])
            if len(suggests) != 0:
                spell_list.extend(suggests)
    if len(spell_list) != 0:
        tokens.extend(spell_list)
    return tokens


def _parse_image(src_folder, dst_folder=None):
    pid = os.getpid()
    p = psutil.Process(pid)
    logging.info('Process id[{}] name[{}]'.format(pid, p.name()))
    images = tools.get_image_files(src_folder)
    logging.info('{} files in [{}]'.format(len(images), src_folder))

    if dst_folder is None:
        dst_folder = src_folder

    for image in images:
        new_name = os.path.splitext(os.path.basename(image))[0]
        new_name = dst_folder + os.sep + new_name + '.txt'
        with open(new_name, 'w') as f:
            image = Image.open(image)
            tokens = _process_text(pytesseract.image_to_string(image))
            f.write(' '.join(tokens).encode('utf8'))


if __name__ == '__main__':
    tools.init_log('./tesseract_image.log')

    # testing 1
    _test1()

