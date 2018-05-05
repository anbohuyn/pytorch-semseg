import argparse
import numpy as np
import os
from PIL import Image

def calculate_mean(args):
    
    images_path = os.path.join(args.dataset,  'data_semantics', 'custom', args.split, 'image_2')
    
    print("Image base {}: ".format(images_path))

    for root, dirs, files in os.walk(images_path):  
        
        n = len(files)
        print("Number of images : {}".format(n))        
        
        images = np.zeros((n, 375, 1242, 3))
        i=0
        for filename in files:
            img = Image.open(os.path.join(images_path,filename))
            a = np.asarray(img, dtype="uint8") # a is readonly
            print("{} {}".format(filename, a.shape)) 
            images[i, :,:,:] = np.clip(a,0,255)
            i += 1

        print images.mean(axis=(0,1,2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--split', nargs='?', type=str, default='train', 
                        help='Split to use [\'train, test, val etc\']')
    args = parser.parse_args()
    calculate_mean(args)
