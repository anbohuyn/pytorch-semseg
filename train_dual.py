import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations_dual import *
#from ptsemseg.augmentations import *

def train(args):

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),
                       RandomHorizontallyFlip(),
                       RandomRotate(-15),                                         
                       RandomRotate(15),        
                       RandomRotate(-10),        
                       RandomCrop(size=256, padding=0),                                         
                       CenterCrop(size=256), 
                       HorizontallyFlip(),
                       RandomSized(size=128),
                       RandomSized(size=256), 
                       RandomSizedCrop(size=256),
                    ])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=False)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    print('Arguments : {}'.format(args))
    model = get_model(args.arch, n_classes)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    print("Using {} GPUs...".format(range(torch.cuda.device_count())))
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    #else:
    #    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=5e-4)


    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"                    
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 

    best_iou = -100.0 
    for epoch in range(args.n_epoch):
        model.train()
        for i, (images, flows, labels) in enumerate(trainloader):
            #images = Variable(images.cuda())
            #flows = Variable(flows.cuda())
            #print("Train images size : {}".format(images.size()))
            #print("Flow images size : {}".format(flows.size()))
            
            inputs_combined = torch.cat([images, flows]) 
            inputs_combined = Variable(inputs_combined.cuda())
            #print('Combined input shape: {}'.format(inputs_combined.size()) )

            labels = Variable(labels.cuda())
            optimizer.zero_grad()
            
            outputs = model(inputs_combined)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')

            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data[0]))

        model.eval()
        for i_val, (images_val, flows_val, labels_val) in tqdm(enumerate(valloader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            flows_val = Variable(flows_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            inputs_combined_val = torch.cat([images_val, flows_val])
            outputs = model(inputs_combined_val)

            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}_{}_best_model.pkl".format(args.arch, args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1, 
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-5, 
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--visdom', nargs='?', type=bool, default=False, 
                        help='Show visualization(s) on visdom | False by  default')
    args = parser.parse_args()
    train(args)