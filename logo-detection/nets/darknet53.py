from functools import wraps
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import cobine_data


@wraps(Conv2D)
def Darknet_basic_Conv2D(*args, **kwargs):
    keys_dict = {}
    keys_dict['kernel_regularizer'] = l2(0.00001)
    if kwargs.get('strides')==(2,2):
        keys_dict['padding'] = 'valid'
    else:
        keys_dict['padding'] = 'same'
    keys_dict.update(kwargs)
    return Conv2D(*args, **keys_dict)

def Darknet_Conv2D_BatchNormal_Leaky_Relu(*args, **kwargs):
    keys_dict = {}
    keys_dict['use_bias'] = False
    keys_dict.update(kwargs)
    return cobine_data(
        Darknet_basic_Conv2D(*args, **keys_dict),
        BatchNormalization(),
        LeakyReLU(0.1))

def res_darknet_block(out, channels_num, num_layers):
    out = ZeroPadding2D(((1, 0),(1, 0)))(out)
    out = Darknet_Conv2D_BatchNormal_Leaky_Relu(channels_num, (3, 3), strides=(2, 2))(out)
    for num in range(num_layers):
        y_out = Darknet_Conv2D_BatchNormal_Leaky_Relu(channels_num//2, (1,1))(out)
        y_out = Darknet_Conv2D_BatchNormal_Leaky_Relu(channels_num, (3,3))(y_out)
        out = Add()([out,y_out])
    return out

def darknet53_bodys(x):
    out = Darknet_Conv2D_BatchNormal_Leaky_Relu(32, (3,3))(x)
    out = res_darknet_block(out, 64, 1)
    out = res_darknet_block(out, 128, 2)
    out = res_darknet_block(out, 256, 8)
    feat_11 = out
    out = res_darknet_block(out, 512, 8)
    feat_22 = out
    out = res_darknet_block(out, 1024, 4)
    feat_33 = out
    return feat_11,feat_22,feat_33

