import glob
import os
import pickle
from PIL import Image
from feature_extractor import FeatureExtractor
import argparse
import concurrent.futures

processor_num = 4
img_types = '.png'

if __name__ == '__main__':
    
    # use to generate features from pretrained neural network and save into pickle file
    
    # eg: python3 generator.py -i "./images/" -f "./mobilenet_features/" -m "nasnetlarge" -s 224

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_root', help='path to the images root')
    parser.add_argument('-f', '--feature_root', help='path to the feature root')
    parser.add_argument('-m', '--model', help='model name', default='mobilenet')
    parser.add_argument('-s', '--image_size', help='image size with equal width and height. eg: 224', default=224, type=int)
    args = parser.parse_args()

    assert(type(args.images_root) == str)
    assert(type(args.feature_root) == str)
    assert(type(args.image_size) == int)

    fe = FeatureExtractor(args.model, args.image_size)

    for root, dirs, files in os.walk(args.images_root):
        datas = [ fi for fi in files if fi.endswith(img_types)]
        for data in datas:
            with concurrent.futures.ProcessPoolExecutor(max_workers=processor_num) as executor:

                try:
                    print('processing...{}'.format(data))

                    img_name,ext = data.rsplit('.',1)
                    img = Image.open(os.path.join(root, data))  # PIL image
                    feature = fe.extract(img)

                    dest_path = os.path.join(args.feature_root, os.path.relpath(root, start=args.images_root))
                    if not os.path.exists(dest_path):
                        os.makedirs(dest_path)

                    feature_path = os.path.join(dest_path, "{}.pkl".format(img_name))
                    pickle.dump(feature, open(feature_path, 'wb'))
                except Exception as e:
                    print(e)
