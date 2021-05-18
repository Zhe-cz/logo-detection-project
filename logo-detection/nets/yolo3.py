from functools import wraps


import tensorflow as tf
from keras import backend as ks
from keras.layers import Conv2D,  UpSampling2D, Concatenate, AveragePooling2D,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from nets.darknet53 import darknet53_bodys
from utils.utils import cobine_data,Image
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization



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

def make_bodys_layers(input, number_filter, output_filter):

    out_x = Darknet_Conv2D_BatchNormal_Leaky_Relu(number_filter, (1,1))(input)
    out_x = Darknet_Conv2D_BatchNormal_Leaky_Relu(number_filter*2, (3,3))(out_x)
    out_x = Darknet_Conv2D_BatchNormal_Leaky_Relu(number_filter, (1,1))(out_x)
    out_x = Darknet_Conv2D_BatchNormal_Leaky_Relu(number_filter*2, (3,3))(out_x)
    out_x = Darknet_Conv2D_BatchNormal_Leaky_Relu(number_filter, (1,1))(out_x)


    out_y = Darknet_Conv2D_BatchNormal_Leaky_Relu(number_filter*2, (3,3))(out_x)
    out_y = Darknet_basic_Conv2D(output_filter, (1,1))(out_y)
            
    return out_x, out_y


def networks_yolo(in_data_shape, number_anchors, number_classes):

    feat_11,feat_22,feat_33 = darknet53_bodys(in_data_shape)
    darknet53_network = Model(in_data_shape, feat_33)


    x_out, y1_out = make_bodys_layers(darknet53_network.output, 512, number_anchors*(number_classes+5))

    x_out = cobine_data(
            Darknet_Conv2D_BatchNormal_Leaky_Relu(256, (1, 1)),
            UpSampling2D(2))(x_out)
    x_out = Concatenate()([x_out, feat_22])

    x_out, y2_out = make_bodys_layers(x_out, 256, number_anchors*(number_classes+5))

    x_out = cobine_data(
            Darknet_Conv2D_BatchNormal_Leaky_Relu(128, (1,1)),
            UpSampling2D(2))(x_out)
    x_out = Concatenate()([x_out,feat_11])

    x_out, y3_out = make_bodys_layers(x_out, 128, number_anchors*(number_classes+5))

    return Model(in_data_shape, [y1_out,y2_out,y3_out])


def yolo_head(feat_input, anchors_box, number_classes, one_channels_shape, jisuan_total_loss=False):
    number_a = len(anchors_box)

    anchors_tensors = ks.reshape(ks.constant(anchors_box), [1, 1, 1, number_a, 2])

    grids_shapes = ks.shape(feat_input)[1:3]
    grids_y_shapes = ks.tile(ks.reshape(ks.arange(0, stop=grids_shapes[0]), [-1, 1, 1, 1]),
                             [1, grids_shapes[1], 1, 1])
    grids_x_shapes = ks.tile(ks.reshape(ks.arange(0, stop=grids_shapes[1]), [1, -1, 1, 1]),
                             [grids_shapes[0], 1, 1, 1])
    grids = ks.concatenate([grids_x_shapes, grids_y_shapes])
    grids = ks.cast(grids, ks.dtype(feat_input))

    f_data = ks.reshape(feat_input, [-1, grids_shapes[0], grids_shapes[1], number_a, number_classes + 5])

    boundingbox_xy_data = (ks.sigmoid(f_data[..., :2]) + grids) / ks.cast(grids_shapes[::-1], ks.dtype(f_data))
    boundingbox_wh_data = ks.exp(f_data[..., 2:4]) * anchors_tensors / ks.cast(one_channels_shape[::-1],
                                                                               ks.dtype(f_data))
    boundingbox_con_score = ks.sigmoid(f_data[..., 4:5])
    boundingbox_class_data = ks.sigmoid(f_data[..., 5:])

    if jisuan_total_loss == True:
        return grids, f_data, boundingbox_xy_data, boundingbox_wh_data
    return boundingbox_xy_data, boundingbox_wh_data, boundingbox_con_score, boundingbox_class_data


def yolo_correct_layer_boxes(boundingb_xx_data, boundingb_ww_data, inputs_shapes, images_shapes):
    boundingb_yy_data = boundingb_xx_data[..., ::-1]
    boundingb_hh_data = boundingb_ww_data[..., ::-1]

    inputs_shapes = ks.cast(inputs_shapes, ks.dtype(boundingb_yy_data))
    images_shapes = ks.cast(images_shapes, ks.dtype(boundingb_yy_data))

    news_shapes = ks.round(images_shapes * ks.min(inputs_shapes / images_shapes))
    offsets_value = (inputs_shapes - news_shapes) / 2. / inputs_shapes
    scales_value = inputs_shapes / news_shapes

    boxs_yx_data = (boundingb_yy_data - offsets_value) * scales_value
    boundingb_hh_data *= scales_value

    boxs_min_iou = boxs_yx_data - (boundingb_hh_data / 2.)
    boxs_max_iou = boxs_yx_data + (boundingb_hh_data / 2.)
    boundingboxs_total = ks.concatenate([
        boxs_min_iou[..., 0:1],
        boxs_min_iou[..., 1:2],
        boxs_max_iou[..., 0:1],
        boxs_max_iou[..., 1:2]
    ])

    boundingboxs_total *= ks.concatenate([images_shapes, images_shapes])
    return boundingboxs_total


#修改到这里
def yolo_boundingboxs_value(feat_box, bounding_box, number_labels, inputs_shapes, images_shapes):

    boxs_xy_data, boxs_wh_data, boxs_confidence_score, boxs_class_probs_score = yolo_head(feat_box, bounding_box, number_labels, inputs_shapes)

    bounding_box_total = yolo_correct_layer_boxes(boxs_xy_data, boxs_wh_data, inputs_shapes, images_shapes)

    bounding_box_total = ks.reshape(bounding_box_total, [-1, 4])
    bounding_box_value = boxs_confidence_score * boxs_class_probs_score
    bounding_box_value = ks.reshape(bounding_box_value, [-1, number_labels])
    return bounding_box_total, bounding_box_value


#改到这里
def yolo_test(ff_input,
              bounding_box_a,
              number_labels,
              images_shapes,
              num_boundingb=20,
              values=.6,
              iou_vaule=.5):

    total_num_ff = len(ff_input)

    anchors_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    inputs_shapes = ks.shape(ff_input[0])[1:3] * 32
    bounding_box_total = []
    bounding_box_total_value = []

    for i in range(total_num_ff):
        _boundingb, _boundingb_value = yolo_boundingboxs_value(ff_input[i], bounding_box_a[anchors_masks[i]], number_labels, inputs_shapes,
                                                    images_shapes)
        bounding_box_total.append(_boundingb)
        bounding_box_total_value.append(_boundingb_value)

    boundingbs_total = ks.concatenate(bounding_box_total, axis=0)
    bounding_box_total_value = ks.concatenate(bounding_box_total_value, axis=0)

    masks_boundingb = bounding_box_total_value >= values
    boundingb_max_T = ks.constant(num_boundingb, dtype='int32')
    boundingbs_ss = []
    values_ss = []
    labels_ss = []
    for x in range(number_labels):

        labels_boundingbs_total = tf.boolean_mask(boundingbs_total, masks_boundingb[:, x])
        labels_boundingbs_value_toatl = tf.boolean_mask(bounding_box_total_value[:, x], masks_boundingb[:, x])


        nms_index = tf.image.non_max_suppression(
            labels_boundingbs_total, labels_boundingbs_value_toatl, boundingb_max_T, iou_threshold=iou_vaule)


        labels_boundingbs_total = ks.gather(labels_boundingbs_total, nms_index)
        labels_boundingbs_value_toatl = ks.gather(labels_boundingbs_value_toatl, nms_index)
        labels_num = ks.ones_like(labels_boundingbs_value_toatl, 'int32') * x
        boundingbs_ss.append(labels_boundingbs_total)
        values_ss.append(labels_boundingbs_value_toatl)
        labels_ss.append(labels_num)
    boundingbs_ss = ks.concatenate(boundingbs_ss, axis=0)
    values_ss = ks.concatenate(values_ss, axis=0)
    labels_ss = ks.concatenate(labels_ss, axis=0)

    return boundingbs_ss, values_ss, labels_ss





