import torch.nn as nn
from torch import chunk

from ptsemseg.models.utils import *

class segnet_flow(nn.Module):

    def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
        super(segnet_flow, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.n_classes = n_classes

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.down1_flow = segnetDown2(self.in_channels, 64)
        self.down2_flow = segnetDown2(64, 128)
        self.down3_flow = segnetDown3(128, 256)
        self.down4_flow = segnetDown3(256, 512)
        self.down5_flow = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def encode_segnet(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
    
        indices = [indices_1, indices_2, indices_3, indices_4, indices_5]
        unpool_shape = [unpool_shape1, unpool_shape2, unpool_shape3, unpool_shape4, unpool_shape5]
        return down5, indices, unpool_shape

    def encode_segnet_flow(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1_flow(inputs)
        down2, indices_2, unpool_shape2 = self.down2_flow(down1)
        down3, indices_3, unpool_shape3 = self.down3_flow(down2)
        down4, indices_4, unpool_shape4 = self.down4_flow(down3)
        down5, indices_5, unpool_shape5 = self.down5_flow(down4)
    
        #indices = [indices_1, indices_2, indices_3, indices_4, indices_5]
        #unpool_shape = [unpool_shape1, unpool_shape2, unpool_shape3, unpool_shape4, unpool_shape5]
        return down5 #, indices, unpool_shape

    def decode_segnet(self, inputs, indices, unpool_shape):

        [indices_1, indices_2, indices_3, indices_4, indices_5] = indices
        [unpool_shape1, unpool_shape2, unpool_shape3, unpool_shape4, unpool_shape5] = unpool_shape

        down5 = inputs
        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        
        return up1

    def forward(self, inputs):
        (images, flows) = inputs

        images_encoded, images_indices, images_unpool_shape  = encode_segnet(images)
        flow_encoded, flow_indices, flow_unpool_shape = encode_segnet_flow(flows)

        weight_flow = Variable(torch.rand(1), requires_grad=True)
        flow_encoded_scaled = flow_encoded * weight_flow.expand_as(flow_encoded)

        inputs_encoded = torch.add(images_encodoed, flow_encoded_scaled) 
        #inputs_encoded = torch.cat((images_encodoed, flow_encodoed), 1)
    
        output = decode_segnet(self, inputs_encoded, images_indices, images_unpool_shape)
        return output

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1,
                  self.down2,
                  self.down3,
                  self.down4,
                  self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit]
            else:
                units = [conv_block.conv1.cbr_unit,
                         conv_block.conv2.cbr_unit,
                         conv_block.conv3.cbr_unit]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
