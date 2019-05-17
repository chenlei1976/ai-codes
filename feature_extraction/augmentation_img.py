import cv2
import numpy as np
import sys, os, shutil
from argparse import ArgumentParser
import multiprocessing as mp
import concurrent.futures

class SimpleAugmentation:

    def __init__(self, *, width_shift_range = 0, height_shift_range = 0, horizontal_flip = False, color_shift_range = 0, gamma_range = 0, rotation = 0, scale = 1, to_gray = False, rand_color = False):

        assert(gamma_range >= 0 and gamma_range < 1)
        assert(width_shift_range >= 0 and height_shift_range >= 0)
        assert(color_shift_range >= 0)

        self.horizontal_flip = horizontal_flip
        self.color_shift_range = int(round(color_shift_range))
        self.width_shift_range = int(round(width_shift_range))
        self.height_shift_range = int(round(height_shift_range))
        self.gamma_range = gamma_range
        self.rotation = rotation
        self.scale = scale
        self.to_gray = to_gray
        self.rand_color = rand_color

    def translate(self, img, tx, ty):
        height, width = img.shape[:2]
        M = np.float32([[1,0,tx], [0,1,ty]])
        return cv2.warpAffine(img, M, (width, height))

    def rotate(self, img, angle, scale = 1):
        height, width = img.shape[:2]
        center = (height/2, width/2)
        rotMat = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(img, rotMat, (width, height))

    def h_flip(self, img):
        return cv2.flip(img, 1)

    def adjust_gamma(self, img, gamma = 1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        return cv2.LUT(img, table)
        
    # more augmentation
    def rand_brightness(self, img):
        img = img.astype(np.float32)
        min_v, max_v = np.min(img), np.max(img)
        delta = np.random.uniform(-5.0, 5.0)
        img += delta
        img = np.clip(img, min_v, max_v)
        return img.astype('uint8')
        
    def rand_blur(self, img):
        img = cv2.bilateralFilter(img,5,15,15)       
        return img.astype('uint8')
        
    def to_rand_color(self, img):
        for i in range(3):
            channel = np.random.randint(3)
            img[:,:,i] = img[:,:,channel]        
        return img.astype('uint8')
        
    def convert2gray(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    def augment(self, img):
        if not type(img) is np.ndarray:
            img = np.array(img)
        img = img.copy()

        h_flip_flag = bool(np.random.randint(2))
        if self.horizontal_flip and h_flip_flag:            
            img = self.h_flip(img)

        if self.color_shift_range > 0:
            channel = np.random.randint(3)
            shift_value = np.random.uniform(-self.color_shift_range, self.color_shift_range)
            img = img.astype(np.float32)
            min_v, max_v = np.min(img), np.max(img)
            img[:,:,channel] = img[:,:,channel] + shift_value
            img = np.clip(img, min_v, max_v)
            img = img.astype('uint8')
            
        #random color
        work = bool(np.random.randint(2))
        if work and self.rand_color:   
            img = self.to_rand_color(img)

        if self.gamma_range > 0:
            min_gamma = 1.0 - self.gamma_range
            max_gamma = 1.0 + self.gamma_range
            gamma = np.random.uniform(min_gamma, max_gamma)
            
            img = self.adjust_gamma(img, gamma)
            
            work = bool(np.random.randint(2))
            if work:    img = self.rand_brightness(img)
                
            work = bool(np.random.randint(2))
            if work:    img = self.rand_blur(img)
                
        to_gray_flag = bool(np.random.randint(2))
		if self.to_gray and to_gray_flag:
			img = self.convert2gray(img)

        if self.rotation > 0 or self.scale != 1:
            rot_val = np.random.uniform(-self.rotation, self.rotation)
            min_scale = 1 - self.scale
            max_scale = 1 + self.scale
            scale_val = np.random.uniform(min_scale, max_scale)

            img = img.astype(np.float32)
            img = self.rotate(img, rot_val, scale_val)
            img = img.astype('uint8')

        if self.width_shift_range > 0 or self.height_shift_range > 0:
            shift_x = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            shift_y = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            
            img = self.translate(img, shift_x, shift_y)

        return img
        
def change_setting(args, folder_basename):

    augHelper = SimpleAugmentation(
                    horizontal_flip = args.hflip,
                    gamma_range = args.gamma_range,
                    color_shift_range = args.color_shift_range,
                    width_shift_range = args.width_shift_range,
                    height_shift_range = args.height_shift_range,
                    rotation = args.rotation,
                    scale = args.scale,
                    to_gray = args.gray,
                    rand_color = args.rand_color
                    )
                    
    return augHelper
        
def multi_process(variation, output, rel_dir, f, augHelper):
            
    for var in range(variation):
        
        out_root = os.path.join(output, rel_dir)
        if rel_dir == '.':
            out_root = output
        if not os.path.exists(out_root):
            os.makedirs(out_root)
        
        aug_id = np.random.randint(1000)        
        filename = str(aug_id) + '_' + os.path.basename(f)
        
        outpath = os.path.join(out_root, filename)
        if os.path.exists(outpath):
            continue
        
        img = cv2.imread(f)        
        img = augHelper.augment(img)
        
        cv2.imwrite(outpath, img)
    
if __name__ == '__main__':

    ap = ArgumentParser()

    ap.add_argument('-i', '--input', type=str, help='root directory', required=True)
    ap.add_argument('-o', '--output', type=str, help='output root directory', required=True)
    ap.add_argument('-hf', '--hflip', help='horizontal flip', action='store_true')
    ap.add_argument('-tx', '--width_shift_range', type=int, help='width shift range (positive intenger)', default = 0)
    ap.add_argument('-ty', '--height_shift_range', type=int, help='height shift range (positive intenger)', default = 0)
    ap.add_argument('-g', '--gamma_range', type=float, help='gamma range (float)', default = 0)
    ap.add_argument('-cs', '--color_shift_range', type=int, help='color shift range', default = 0)
    ap.add_argument('-v', '--var', type=int, help='augmentation variation', default = 1)
    ap.add_argument('-rot', '--rotation', type=int, help='rotation angle in degree', default = 0)
    ap.add_argument('-gr', '--gray', action='store_true', default = False)
    ap.add_argument('-rc', '--rand_color', action='store_true', default = False)
    ap.add_argument('-sc', '--scale', type=float, help='scale, which apply when performing rotation', default=1)
    ap.add_argument('-wk', '--workers', type=int, help='number of workers', default = 4)

    args = ap.parse_args()

    output = args.output
    workers = args.workers
    
    variation = args.var
    
    img_types = ['.jpg','.png']

    for root, dirs, files in os.walk(args.input, topdown = False):
        filelist = [ os.path.join(root,fi) for fi in files if fi.endswith(tuple(img_types))]
        
        rel_dir = os.path.relpath(root, start=args.input)
        
        if len(filelist) == 0: continue
                
        folder_basename = os.path.basename(root)
        print('processing...',folder_basename)
        augHelper = change_setting(args, folder_basename)
        
        for f in filelist:
        
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                multi_process(variation, output, rel_dir, f, augHelper)
            