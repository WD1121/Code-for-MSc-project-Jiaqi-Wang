import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

print('torch_version: ', torch.__version__)
print('torchvision_version: ', torchvision.__version__)

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # Residual part
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # Replace in place to save memory cost
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        # Because there are inconsistent dimensions, it is divided into different situations
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet18(nn.Module):
    def __init__(self, BasicBlock, num_classes=10) -> None:
        super(resnet18, self).__init__()
        self.in_channels = 64
        # The first layer is treated as a separate layer and there is no residual block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])

        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])

        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])

        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    #This function is mainly used to repeat the same residual block
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        # out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

#ResNet18 = resnet18(BasicBlock)
#print(ResNet18)
def ResNet18():
		return resnet18(BasicBlock)







#
