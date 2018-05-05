# pytorch-semseg

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/pytorch-semseg/blob/master/LICENSE)
[![pypi](https://img.shields.io/pypi/v/pytorch_semseg.svg)](https://pypi.python.org/pypi/pytorch-semseg/0.1.2)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1185075.svg)](https://doi.org/10.5281/zenodo.1185075)

## Semantic Segmentation [PyTorch] for Autonomous Driving Scenes

Adapted from https://github.com/meetshah1995/pytorch-semseg

<p align="center">
<a href="https://www.youtube.com/watch?v=iXh9aCK3ubs" target="_blank"><img src="https://i.imgur.com/agvJOPF.gif" width="364"/></a>
<img src="https://meetshah1995.github.io/images/blog/ss/ptsemseg.png" width="49%"/>
</p>


### Networks implemented

* [FCN](https://arxiv.org/abs/1411.4038) - All 1 (FCN32s), 2 (FCN16s) and 3 (FCN8s) stream variants
* [Segnet](https://arxiv.org/abs/1511.00561) - With Unpooling using Maxpool indices
* [PSPNet](https://arxiv.org/abs/1612.01105) - With support for loading pretrained models w/o caffe dependency


### Datasets/Dataloaders

* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)


### Requirements

* pytorch >=0.3.0
* torchvision ==0.2.0
* visdom >=1.0.1 (for loss and results visualization)
* scipy
* tqdm

#### One-line installation
    
`pip install -r requirements.txt`


**To train the model :**

```
python train.py [-h] [--arch [ARCH]] [--dataset [DATASET]]
                [--img_rows [IMG_ROWS]] [--img_cols [IMG_COLS]]
                [--n_epoch [N_EPOCH]] [--batch_size [BATCH_SIZE]]
                [--l_rate [L_RATE]] [--feature_scale [FEATURE_SCALE]]
                [--visdom [VISDOM]]

  --arch           Architecture to use ['fcn8s, unet, segnet etc']
  --dataset        Dataset to use ['pascal, camvid, ade20k etc']
  --img_rows       Height of the input image
  --img_cols       Width of the input image
  --n_epoch        # of the epochs
  --batch_size     Batch Size
  --l_rate         Learning Rate
  --feature_scale  Divider for # of features to use
  --visdom         Show visualization(s) on visdom | False by default
```

**To validate the model :**

```
python validate.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
                   [--img_rows [IMG_ROWS]] [--img_cols [IMG_COLS]]
                   [--batch_size [BATCH_SIZE]] [--split [SPLIT]]

  --model_path   Path to the saved model
  --dataset      Dataset to use ['pascal, camvid, ade20k etc']
  --img_rows     Height of the input image
  --img_cols     Width of the input image
  --batch_size   Batch Size
  --split        Split of dataset to validate on
```

**To test the model w.r.t. a dataset on custom images(s):**

```
python test.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
               [--dcrf [DCRF]] [--img_path [IMG_PATH]] [--out_path [OUT_PATH]]
 
  --model_path          Path to the saved model
  --dataset             Dataset to use ['pascal, camvid, ade20k etc']
  --dcrf                Enable DenseCRF based post-processing
  --img_path            Path of the input image
  --out_path            Path of the output segmap
```


**Reference**

```
@article{mshahsemseg,
    Author = {Meet P Shah},
    Title = {Semantic Segmentation Architectures Implemented in PyTorch.},
    Journal = {https://github.com/meetshah1995/pytorch-semseg},
    Year = {2017}
}
```

