import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import scipy.misc as misc
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

from os import walk, path

try:
    import pydensecrf.densecrf as dcrf
except:
    print("Failed to import pydensecrf,\
           CRF post-processing will not work")

def test_all_dual(args):

    root = args.dir_path
    flow_root = args.flow_path    
    out = args.dir_out_path

    files = []
    for (dirpath, dirnames, filenames) in walk(root):
        files.extend(filenames)
        break

    for f in sorted(files):
        args.img_path = path.join(root, f)
        args.flow_path = path.join(flow_root, f)

        filename_noext, ext = path.splitext(f)
        new_filename = '{}_{}{}'.format(filename_noext, 'out' , ext)
        args.out_path = path.join(out, new_filename)
        test_one(args) 

def test_one(args):
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))
    img = misc.imread(args.img_path)
    flow = misc.imread(args.flow_path)
    print("Read Flow Image from : {}".format(args.flow_path))
    
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True)
    n_classes = loader.n_classes
    
    #Image
    resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp='bicubic')

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))
    img = img.astype(float) / 255.0
    # NHWC -> NCWH
    img = img.transpose(2, 0, 1) 
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    #Flow
    #resized_flow = misc.imresize(flow, (self.img_size[0], self.img_size[1]))
    resized_flow = misc.imresize(flow, (loader.img_size[0], loader.img_size[1]))
    
    flow = flow.astype(np.float64)
    flow = misc.imresize(flow, (loader.img_size[0], loader.img_size[1]))
    flow = flow.transpose(2, 0, 1)
    flow = np.expand_dims(flow, 0)
    flow = torch.from_numpy(flow).float()

    # Setup Model
    model = get_model(args.model_path[:args.model_path.find('_')], n_classes)
    state = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    
    model.cuda(0)
    images = Variable(img.cuda(0), volatile=True)
    flows = Variable(flow.cuda(0), volatile=True)

    inputs_combined = torch.cat([images, flows]) 

    outputs = F.softmax(model(inputs_combined), dim=1)
    
    if args.dcrf == "True":
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)
       
        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = args.out_path[:-4] + '_drf.png'
        misc.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)
    print('Classes found: ', np.unique(pred))
    misc.imsave(args.out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(args.out_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', nargs='?', type=str, default='fcn8s_pascal_1_26.pkl', 
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--dcrf', nargs='?', type=str, default="False",
                        help='Enable DenseCRF based post-processing')
    parser.add_argument('--dir_path', nargs='?', type=str, default=None, 
                        help='Directory Path of the input images')
    parser.add_argument('--flow_path', nargs='?', type=str, default=None, 
                        help='Directory Path of the flow images')
    parser.add_argument('--dir_out_path', nargs='?', type=str, default=None, 
                        help='Directory Path of the output segmaps')
    args = parser.parse_args()
    test_all_dual(args)
