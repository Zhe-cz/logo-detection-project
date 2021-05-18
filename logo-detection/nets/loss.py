import numpy as np
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as ks
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.regularizers import l2


def yolo_head_loss(feat_nums, anchors_box, total_class, one_channels_shape, jisuan_total_loss=False):
    number_a = len(anchors_box)

    anchors_tensors = ks.reshape(ks.constant(anchors_box), [1, 1, 1, number_a, 2])


    grids_shapes = ks.shape(feat_nums)[1:3]
    grids_y_shapes = ks.tile(ks.reshape(ks.arange(0, stop=grids_shapes[0]), [-1, 1, 1, 1]),
                    [1, grids_shapes[1], 1, 1])
    grids_x_shapes = ks.tile(ks.reshape(ks.arange(0, stop=grids_shapes[1]), [1, -1, 1, 1]),
                    [grids_shapes[0], 1, 1, 1])
    grids = ks.concatenate([grids_x_shapes, grids_y_shapes])
    grids = ks.cast(grids, ks.dtype(feat_nums))


    f_data = ks.reshape(feat_nums, [-1, grids_shapes[0], grids_shapes[1], number_a, total_class + 5])


    boundingbox_xy_data = (ks.sigmoid(f_data[..., :2]) + grids) / ks.cast(grids_shapes[::-1], ks.dtype(f_data))
    boundingbox_wh_data = ks.exp(f_data[..., 2:4]) * anchors_tensors / ks.cast(one_channels_shape[::-1], ks.dtype(f_data))
    boundingbox_con_score = ks.sigmoid(f_data[..., 4:5])
    boundingbox_class_data = ks.sigmoid(f_data[..., 5:])

    if jisuan_total_loss == True:
        return grids, f_data, boundingbox_xy_data, boundingbox_wh_data
    return boundingbox_xy_data, boundingbox_wh_data, boundingbox_con_score, boundingbox_class_data



def boxs_iou(b1_boundingbox, b2_boundingbox):

    bounding1_box = ks.expand_dims(b1_boundingbox, -2)
    bounding1_xy_data = bounding1_box[..., :2]
    bounding1_wh_data = bounding1_box[..., 2:4]
    bounding1_wh_part_data = bounding1_wh_data / 2.
    bounding1_min_data = bounding1_xy_data - bounding1_wh_part_data
    bounding1_max_data = bounding1_xy_data + bounding1_wh_part_data


    bounding2_box = ks.expand_dims(b2_boundingbox, 0)
    bounding2_xy_data = bounding2_box[..., :2]
    bounding2_wh_data = bounding2_box[..., 2:4]
    bounding2_wh_part_data = bounding2_wh_data / 2.
    bounding2_min_data = bounding2_xy_data - bounding2_wh_part_data
    bounding2_max_data = bounding2_xy_data + bounding2_wh_part_data


    p_min_data = ks.maximum(bounding1_min_data, bounding2_min_data)
    p_max_data = ks.minimum(bounding1_max_data, bounding2_max_data)
    p_wh_data = ks.maximum(p_max_data - p_min_data, 0.)
    p_total = p_wh_data[..., 0] * p_wh_data[..., 1]
    bounding1_a_data = bounding1_wh_data[..., 0] * bounding1_wh_data[..., 1]
    bounding2_a_data = bounding2_wh_data[..., 0] * bounding2_wh_data[..., 1]
    ious_boxs = p_total / (bounding1_a_data + bounding2_a_data - p_total)

    return ious_boxs



def yolos_total_losss(kywags, anchors, number_labels, i_value=.5, count_losss=False):
    number_network = len(anchors) // 3


    y_labels_data = kywags[number_network:]
    network_outputs_data = kywags[:number_network]
    if number_network == 3 :
        anchors_layer_m = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    else:
        anchors_layer_m = [[3, 4, 5], [1, 2, 3]]

    one_channels_shape = ks.cast(ks.shape(network_outputs_data[0])[1:3] * 32, ks.dtype(y_labels_data[0]))


    grids_layer_s = [ks.cast(ks.shape(network_outputs_data[l])[1:3], ks.dtype(y_labels_data[0])) for l in range(number_network)]
    losses = 0


    temp = ks.shape(network_outputs_data[0])[0]
    temp_i = ks.cast(temp, ks.dtype(network_outputs_data[0]))


    for i in range(number_network):

        current_layer_m = y_labels_data[i][..., 4:5]

        true_labels_scores = y_labels_data[i][..., 5:]


        grids, feats_data, box_xy_data, box_wh_data = yolo_head_loss(network_outputs_data[i],
                                                     anchors[anchors_layer_m[i]], number_labels, one_channels_shape, calc_loss=True)


        preds_boxs = ks.concatenate([box_xy_data, box_wh_data])


        currnet_layers_masks = tf.TensorArray(ks.dtype(y_labels_data[0]), dynamic_size=True, size=1)
        current_layer_masks_panduan = ks.cast(current_layer_m, 'bool')


        def wh_bodys(bbox, mask_current):

            ground_boxs = tf.boolean_mask(y_labels_data[i][bbox, ..., 0:4], current_layer_masks_panduan[bbox, ..., 0])

            ious = boxs_iou(preds_boxs[bbox], ground_boxs)


            first_ious = ks.max(ious, axis=-1)


            mask_current = mask_current.write(bbox, ks.cast(first_ious < i_value, ks.dtype(ground_boxs)))
            return bbox + 1, mask_current


        _, currnet_layers_masks = ks.control_flow_ops.while_loop(lambda b, *args: b < temp, wh_bodys, [0, currnet_layers_masks])


        currnet_layers_masks = currnet_layers_masks.stack()

        currnet_layers_masks = ks.expand_dims(currnet_layers_masks, -1)


        bb_label_xx_data = y_labels_data[i][..., :2] * grids_layer_s[i][:] - grids
        bb_label_ww_data = ks.log(y_labels_data[i][..., 2:4] / anchors[anchors_layer_m[i]] * one_channels_shape[::-1])


        bb_label_ww_data = ks.switch(current_layer_m, bb_label_ww_data, ks.zeros_like(bb_label_ww_data))
        boundingb_losss_value = 2 - y_labels_data[i][..., 2:3] * y_labels_data[i][..., 3:4]

        bounding_xx_losses = current_layer_m * boundingb_losss_value * ks.binary_crossentropy(bb_label_xx_data, feats_data[..., 0:2],
                                                                       from_logits=True)
        bounding_ww_losses = current_layer_m * boundingb_losss_value * 0.5 * ks.square(bb_label_ww_data - feats_data[..., 2:4])


        true_loss_total = current_layer_m * ks.binary_crossentropy(current_layer_m, feats_data[..., 4:5], from_logits=True) + \
                          (1 - current_layer_m) * ks.binary_crossentropy(current_layer_m, feats_data[..., 4:5],
                                                                    from_logits=True) * currnet_layers_masks

        label_total_loss = current_layer_m * ks.binary_crossentropy(true_labels_scores, feats_data[..., 5:], from_logits=True)

        bounding_xx_losses = ks.sum(bounding_xx_losses) / temp_i
        bounding_ww_losses = ks.sum(bounding_ww_losses) / temp_i
        true_loss_total = ks.sum(true_loss_total) / temp_i
        label_total_loss = ks.sum(label_total_loss) / temp_i
        losses += bounding_xx_losses + bounding_ww_losses + true_loss_total + label_total_loss
        if count_losss:
            losses = tf.Print(losses, [losses, bounding_xx_losses, bounding_ww_losses, true_loss_total, label_total_loss, ks.sum(currnet_layers_masks)],
                            message='loss: ')
    return losses