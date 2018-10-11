# -*- coding: UTF-8 -*-
import os
import shutil
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

# # google
# google_storage = {'root_dir': '/home/chenlei/images/icrawler'}
# google_crawler = GoogleImageCrawler(parser_threads=4, downloader_threads=4, storage=google_storage)
# google_crawler.crawl(keyword='clinic receipt', max_num=1000)

image_path = '/home/chenlei/images/icrawler/'
classes = ['boarding pass', 'luggage', 'receipt', 'clinic receipt', 'hospital receipt', 'passport first pages', 'broken ipad', 'broken mobile phone']

print(os.path.join('home', "me", "mywork"))

for item in classes:
    img_folder = os.path.join(image_path + item.replace(' ', '_'))
    if not os.path.exists(img_folder):
        print('create ' + img_folder)
        os.mkdir(img_folder)
    bing_storage = {'root_dir': img_folder}
    bing_crawler = BingImageCrawler(parser_threads=2, downloader_threads=4, storage=bing_storage)
    bing_crawler.crawl(keyword=item, max_num=2000)

# #baidu
# baidu_storage = {'root_dir': '/home/chenlei/images/icrawler'}
# baidu_crawler = BaiduImageCrawler(parser_threads=2, downloader_threads=4, storage=baidu_storage)
# baidu_crawler.crawl(keyword='passport', max_num=1000)
