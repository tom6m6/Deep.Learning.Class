import numpy as np

import torch
from torch import nn

def relu(x):
    return np.where(x > 0, x, np.zeros_like(x))


class linear(nn.Module):
    def __init__(self, num_outputs, num_inputs):
        super(linear, self).__init__()
        self.weight = np.random.randn(num_inputs, num_outputs)
        self.bias = np.random.randn(num_outputs)

    def forward_np(self, X):

        Y = np.matmul(X, np.transpose(self.weight)) + np.expand_dims(self.bias, axis=0)
        return Y


def corr2d(X, K):
    '''
    X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
    K --> (O, I, h, w) where O = out_channel, I = in_channel, h = height of kernel, w = width of kernel
    你需要实现一个Stride为1，Padding为0的窄卷积操作
    Y should have size (B, O, H-h+1, W-w+1)
    '''
    # ==========
    # todo '''请完成卷积操作'''
    # ==========   
    B, I, H, W = X.shape
    O, _, h, w = K.shape
    
    Y = np.zeros((B, O, H - h + 1, W - w + 1))
    for b in range(B): # batch size
        for o in range(O): # out_channel
            for t in range(I): # in_channel
                for i in range(H - h + 1): # height(out)
                    for j in range(W - w + 1): # width(out)
                        window = X[b, t, i:i+h, j:j+w] # input window
                        Y[b, o, i, j] += np.sum(window * K[o, t, :, :])

    return Y


class Conv2D(nn.Module):
    def __init__(self, out_channels, in_channels, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = np.random.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.bias = np.random.randn(out_channels)

    def forward_np(self, X):
        '''
        X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
        你需要利用以上初始化的参数weight和bias实现一个卷积层的前向传播
        Y should have size (B, O, H-h+1, W-w+1)
        '''

        # ==========
        # todo '''请完成卷积层的前向传播'''
        # ========== 

        Y = corr2d(X, self.weight)  # corr2d进行卷积
        # bias
        Y += np.reshape(self.bias, (1, -1, 1, 1))

        return Y

class MaxPool2D(nn.Module):
    def __init__(self, pool_size):
        super(MaxPool2D, self).__init__()
        self.pool_size = pool_size


    def forward_np(self, X):
        '''
        X --> (B, I, H, W) where B = batch size, I = in_channel, H = height of feature map, W = width of feature map
        K --> (h, w) where h = height of kernel, w = width of kernel
        你需要利用以上pool_size实现一个汇聚层的前向传播，汇聚层的子区域间无覆盖
        Y should have size (B, I, H/h, W/w)
        '''
        # ==========
        # todo '''请完成最大汇聚层的前向传播'''
        # ========== 
        B, I, H, W = X.shape
        h, w = self.pool_size
        Y = np.zeros((B, I, H // h, W // w))

        for b in range(B):
            for t in range(I):
                for i in range(0, H, h):
                    for j in range(0, W, w):
                        Y[b, t, i//h, j//w] = np.max(X[b, t, i:i+h, j:j+w])

        return Y

class ImageCNN(nn.Module):
    def __init__(self, input_size, num_outputs, in_channels, out_channels, conv1_kernel, pool1_kernel, conv2_kernel, pool2_kernel):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2D(out_channels, in_channels, conv1_kernel),
            nn.ReLU()
        )
        self.pool1 = MaxPool2D(pool1_kernel)

        self.conv2 = nn.Sequential(
            Conv2D(out_channels, in_channels, conv2_kernel),
            nn.ReLU()
        )
        self.pool2 = MaxPool2D(pool2_kernel)

        M, N = input_size
        CK1_L, CK1_W = conv1_kernel
        M1 = (M - CK1_L) + 1
        N1 = (N - CK1_W) + 1
        PK1_L, PK1_W = pool1_kernel
        M2 = int(np.floor((M1 - PK1_L)*1./PK1_L) + 1)
        N2 = int(np.floor((N1 - PK1_W)*1./PK1_W) + 1)

        CK2_L, CK2_W = conv2_kernel
        M3 = (M2 - CK2_L) + 1
        N3 = (N2 - CK2_W) + 1
        PK2_L, PK2_W = pool2_kernel
        M4 = int(np.floor((M3 - PK2_L)*1./PK2_L) + 1)
        N4 = int(np.floor((N3 - PK2_W)*1./PK2_W) + 1)
        self.linear = linear(out_channels*M4*N4, num_outputs)

    def load_state_dict_to_np(self, checkpoint):
        self.conv1[0].weight = checkpoint['conv1.0.weight'].cpu().detach().numpy()
        self.conv1[0].bias = checkpoint['conv1.0.bias'].cpu().detach().numpy()
        self.conv2[0].weight = checkpoint['conv2.0.weight'].cpu().detach().numpy()
        self.conv2[0].bias = checkpoint['conv2.0.bias'].cpu().detach().numpy()
        self.linear.weight = checkpoint['linear.weight'].cpu().detach().numpy()
        self.linear.bias = checkpoint['linear.bias'].cpu().detach().numpy()

    def forward_np(self, feature_map):
        feature_map_np = feature_map.cpu().detach().numpy()
        b = np.shape(feature_map_np)[0]
        feature_map_np = self.conv1[0].forward_np(feature_map_np)
        feature_map_np = relu(feature_map_np)
        feature_map_np = self.pool1.forward_np(feature_map_np)
        feature_map_np = self.conv2[0].forward_np(feature_map_np)
        feature_map_np = relu(feature_map_np)
        feature_map_np = self.pool2.forward_np(feature_map_np)
        outputs = self.linear.forward_np(np.reshape(feature_map_np, (b, -1)))

        return outputs



if __name__ == '__main__':
    model = Model_NP()