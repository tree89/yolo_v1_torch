import torch
import torch.nn as nn

from src.config import architecture_config


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x


class YOLOV1(nn.Module):
    def __init__(self, 
                 in_ch:int=3,
                 split_size=7, 
                 num_boxes=2, 
                 num_classes=20):
        super(YOLOV1, self).__init__()
        self.in_ch = in_ch
        self.darknet = self._get_conv_layers(architecture_config)
        self.fcs = self._get_fcs(split_size=7, num_boxes=2, num_classes=20)
        
    def _get_conv_layers(self, architecture_config):
        layers = []
        in_ch = self.in_ch
                 
        for l in architecture_config:
          if type(l) == tuple:
              layers += [ConvBlock(
                            in_ch, 
                            l[1], 
                            kernel_size=l[0], 
                            stride=l[2], 
                            padding=l[3]
                            )
                        ]
              in_ch = l[1]
          elif type(l) == str:
              layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          elif type(l) == list:
              conv1 = l[0] # tuple
              conv2 = l[1] # tuple
              num_repeats = l[2] # integer
              for _ in range(num_repeats):
                  layers += [ConvBlock(
                                in_ch,
                                conv1[1],
                                kernel_size=conv1[0],
                                stride=conv1[2],
                                padding=conv1[3]
                                )
                            ]

                  layers += [ConvBlock(
                                conv1[1],
                                conv2[1],
                                kernel_size=conv2[0],
                                stride=conv2[2],
                                padding=conv2[3]
                                )
                            ]

                  in_ch = conv2[1]
                  
        return nn.Sequential(*layers)
    
    def _get_fcs(self, split_size, num_boxes, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*split_size*split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, split_size*split_size*(num_classes+num_boxes*5)),  # In pascal voc case -> (S,S,30)
        )