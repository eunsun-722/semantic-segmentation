
# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import math
from skimage import data
from skimage.transform import resize



def transformImage(image, dim=None):
    dimsion = (128,128,128)
    if (dim != None):
        dimension = dim
    return resize(image, dimsion)


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
     #   activation = nn.ReLU(inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]

        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]

        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]

        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]

        down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]

        # Bridge
        bridge = self.bridge(pool_5) # -> [1, 128, 4, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5) # -> [6, 4, 128, 128, 128]
        return out

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        targets = targets.view(4,-1)

        inputs = inputs.view(4,-1)

        intersection = (inputs * targets).sum(1)

        dice = (2.*intersection + smooth)/( (inputs.square()).sum(1) + (targets.square()).sum(1) + smooth)

        return 1 - dice.mean()

def data_select(i):
    if i>=1 and i<10:
        file_name = '../input/train1/00{}_imgs.npy'.format(i)
        file_name2 = '../input/train1/00{}_seg.npy'.format(i)
    if i>=10 and i<100:
        file_name = '../input/train1/0{}_imgs.npy'.format(i)
        file_name2 = '../input/train1/0{}_seg.npy'.format(i)
    if i>=100:
        file_name = '../input/train1/{}_imgs.npy'.format(i)
        file_name2 = '../input/train1/{}_seg.npy'.format(i)
    a = np.load(file_name)
    #a = torchvision.transforms.functional.center_crop(torch.FloatTensor(a),[a[0],a[1]])

    b = np.load(file_name2)
    return a, b

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def adjust_learning_rate(optimizer, e):
    if e ==5:
      for param_group in optimizer.param_groups:
          param_group['lr'] *= 0.5


def interpolate(img, dimension=None):
    #img = img.permute(0, 1, 2)
    dim = (128,160,128)
    if (dimension!=None):
        dim = dimension
    img = torch.nn.functional.interpolate(img, size=dim, mode="trilinear")
    return img

def interpolate2(img, dimension=None):
    #img = img.permute(0, 1, 2)
    img = torch.Tensor(img)
    dim = (128,160,128)
    if (dimension!=None):
        dim = dimension
    img = torch.nn.functional.interpolate(img, size=dim, mode="trilinear")
    return img
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  #print(device)

  model = UNet(in_dim=4, out_dim=4, num_filters=8)           ## num filters 4,8,16,32

  lr = 0.001
  optimizer = optim.Adam(model.parameters(), lr) #SGD? ADAM?
  criterion = DiceLoss()
  num_epochs = 500
  num_data = 204

  batch_size = 6

#  checkpoint = torch.load('../input/777777/model0_net7.npy')
#  checkpoint = torch.load('./model0_net11.npy') #../input/model4/model4_net3.npy
#  model.load_state_dict(checkpoint['model_state_dict'])
#  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # load trained model and retrain it
#  optimizer.param_groups[0]['lr'] = .005 #change learning rate


  optimizer_to(optimizer,device)
  # print(optimizer.param_groups[0]['lr'])
  model.to(device)

  for i in range(num_epochs):
      adjust_learning_rate(optimizer,i)
      average = []
      optimizer.zero_grad()
      for j in range(204):

          a,b = data_select(j+1)

          a = np.expand_dims(a,axis=0)
          a = interpolate2(a)
          a = (torch.squeeze(a, axis=0)).detach().numpy()  #[4,128,160,128]

          H,W,D  = b.shape  #[H,W,D]

          b1_temp = b.copy()
          b2_temp = b.copy()
          b3_temp = b.copy()
          b4_temp = b.copy()
          b_temp = np.zeros((4,H,W,D))

          b1_temp[b1_temp==2.0] = 3.0
          b1_temp[b1_temp==1.0] = 3.0
          b1_temp[b1_temp==0.0] = 1.0
          b1_temp[b1_temp==3.0] = 0.0


          b2_temp[b2_temp==2.0] = 0.0
          b2_temp[b2_temp==3.0] = 0.0

          b3_temp[b3_temp==1.0] = 3.0
          b3_temp[b3_temp==0.0] = 3.0
          b3_temp[b3_temp==2.0] = 1.0
          b3_temp[b3_temp==3.0] = 0.0

          b4_temp[b4_temp==1.0] = 2.0
          b4_temp[b4_temp==3.0] = 1.0
          b4_temp[b4_temp==2.0] = 0.0

          b_temp[0] = b1_temp
          b_temp[1] = b2_temp
          b_temp[2] = b3_temp
          b_temp[3] = b4_temp

          b = torch.Tensor(b_temp)  #[4,H,W,D]
          b.requires_grad = True
          b = b.cuda()


          a = np.expand_dims(a,axis=0)  #[1,4,128,128,128]
          a = torch.Tensor(a)           #convert to Tensor
          a.requires_grad = True
          a = a.cuda()

          x = model(a)             #[1,4,128,128,128]
          x = interpolate(x, (H,W,D))   #[1,4,H,W,D]
          x = (torch.squeeze(x, axis=0)) #[4,H,W,D]
          x = F.softmax(x, dim=0)           #[4,H,W,D]

          loss = criterion(x, b)   ##change b1 b2 b3 b4
          loss.backward()
        #   optimizer.step()
          if ((j+1) % 4 == 0):         ##step every 12th turn(batch size of 12)
              optimizer.step()
              optimizer.zero_grad()




          average.append(loss.item())
      #print(sum(average)/len(average))
      torch.save({

            'model_state_dict': model.state_dict(),                             #change model name
            'optimizer_state_dict': optimizer.state_dict(),


            }, './model{}_net{}.npy'.format(0, i+1))

## function needed for validation check
  out = model(a)
  loss = criterion(out, b)
