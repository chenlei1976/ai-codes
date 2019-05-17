import os
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
import glob
import pickle
import cv2
import concurrent.futures
import argparse

processor_num = 4
img_types = '.png'

def draw_text(img, text):
    img = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    # img = Image.open(img)
    height, width, channels = img.shape
    cv2.rectangle(img, (0,0), (width,30), (0,0,0), -1)
    cv2.putText(img,'{}'.format(text),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    return img

if __name__ == '__main__':

    # use to find similar test image from database features
    
    # eg: python3 server.py -t "./test/1.jpg" -i "./images/" -f "./mobilenet_features/" -r "./predicted/" -m "nasnetlarge" -s 224

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_data', help='path to the test image')
    parser.add_argument('-i', '--images_root', help='path to the all images')
    parser.add_argument('-f', '--feature_root', help='path to the feature root')
    parser.add_argument('-r', '--results_root', help='path to the results')
    parser.add_argument('-m', '--model', help='model name', default='mobilenet')
    parser.add_argument('-s', '--image_size', help='image size with equal width and height. eg: 224', default=224, type=int)
    args = parser.parse_args()
    
    assert(type(args.test_data) == str)
    assert(type(args.images_root) == str)
    assert(type(args.feature_root) == str)
    assert(type(args.results_root) == str)
    assert(type(args.image_size) == int)

    # Read image features
    fe = FeatureExtractor(args.model, args.image_size)
    features = []
    img_paths = []

    for root, dirs, files in os.walk(args.feature_root):
        datas = [ fi for fi in files]
        # print('len',len(datas))
        for data in datas:
            with concurrent.futures.ProcessPoolExecutor(max_workers=processor_num) as executor:
                try:
                    img_name,ext = data.rsplit('.',1)
                    features.append(pickle.load(open(os.path.join(root, data), 'rb')))
                    img_paths.append(os.path.join(args.images_root, os.path.relpath(root, start=args.feature_root), img_name + img_types))
                except Exception as e:
                    print(e)

    img = Image.open(args.test_data)  # PIL image

    query = fe.extract(img)
    print(len(features))
    print(np.shape(features))
    print(np.shape(query))
    dists = np.linalg.norm(features - query, axis=1)  # Do search

    ids = np.argsort(dists)[:10] # Top 30 results
    scores = [(dists[id], img_paths[id]) for id in ids]
    
    dest_path = os.path.join(args.results_root, os.path.splitext(os.path.basename(args.test_data))[0])
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    for score in scores:
        try:
            print('image={}, score={}'.format(score[1],score[0]))
            img = draw_text(score[1], score[0])
            cv2.imwrite(os.path.join(dest_path, os.path.basename(score[1])),img)
        except Exception as e:
            print(e)

