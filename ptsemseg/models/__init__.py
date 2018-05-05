import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.fcn_flow import *
from ptsemseg.models.segnet import *
from ptsemseg.models.segnet_flow import *
from ptsemseg.models.segnet_depth import *
from ptsemseg.models.pspnet import *

def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name in ['fcn32s', 'fcn16s', 'fcn8s']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    
    elif name in ['fcn32s_flow', 'fcn16s_flow', 'fcn8s_flow']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet_flow':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == 'segnet_depth':
        model = model(n_classes=n_classes,
                      is_unpooling=True)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'fcn32s': fcn32s,
            'fcn16s': fcn16s,
            'fcn8s': fcn8s,
            'fcn32s_flow': fcn32s_flow,
            'fcn16s_flow': fcn16s_flow,
            'fcn8s_flow': fcn8s_flow,
            'segnet': segnet,
            'segnet_flow': segnet_flow,
            'segnet_depth': segnet_depth,
            'pspnet': pspnet,
        }[name]
    except exc:
        print exc
        print('Model {} not available'.format(name))
