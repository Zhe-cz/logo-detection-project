import numpy as np
import os
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from nets.yolo3 import networks_yolo
from nets.loss import yolos_total_losss
import keras.backend as K
from keras.layers import Lambda
from keras.optimizers import Adam
from utils.utils import give_rand_dataset
from keras.models import load_model,save_model,model_from_json,clone_model
from keras.layers import Input,core,average
from PIL import Image, ImageFont, ImageDraw, FontFile, SgiImagePlugin
from keras.models import Model
from keras.backend.tensorflow_backend import set_session


def give_total_labels(label_dir):

    with open(label_dir) as file:
        labels_numbers = file.readlines()
    labels_numbers = [x.strip() for x in labels_numbers]
    return labels_numbers

def give_total_boundingbox(ancor_dir):

    with open(ancor_dir) as file:
        ancor_num = file.readline()
    ancor_num = [float(x) for x in ancor_num.split(',')]
    return np.asarray(ancor_num).reshape(-1, 2)

def give_dataset_create(nums, bs, in_shapes, ancors, number_classes):

    num = len(nums)
    x = 0
    while True:
        img_datasets = []
        boxs_dataset = []
        for b in range(bs):
            if x == 0:
                np.random.shuffle(nums)
            imgs, boxs = give_rand_dataset(nums[x], in_shapes, random=True)
            img_datasets.append(imgs)
            boxs_dataset.append(boxs)
            x = (x+1) % num
        img_datasets = np.array(img_datasets)
        boxs_dataset = np.array(boxs_dataset)
        labels = preprocess_true_boxes(boxs_dataset, in_shapes, ancors, number_classes)
        yield [img_datasets, *labels], np.zeros(bs)



def preprocess_true_boxes(ground_boundingbox, in_shapes, ancors, number_labels):

    assert (ground_boundingbox[..., 4]<number_labels).all(), '.'

    number_net = len(ancors)//3
    if number_net ==3 :
        boundingbox_mask = [[6,7,8], [3,4,5], [0,1,2]]
    else:
        boundingbox_mask = [[3,4,5], [1,2,3]]

    ground_boundingboxs = np.array(ground_boundingbox, dtype='float32')
    in_shapes = np.array(in_shapes, dtype='int32') # 416,416

    boundingbox_xx = (ground_boundingboxs[..., 0:2] + ground_boundingboxs[..., 2:4]) // 2
    boundingbox_ww = ground_boundingboxs[..., 2:4] - ground_boundingboxs[..., 0:2]

    ground_boundingboxs[..., 0:2] = boundingbox_xx/in_shapes[::-1]
    ground_boundingboxs[..., 2:4] = boundingbox_ww/in_shapes[::-1]


    m_temp = ground_boundingboxs.shape[0]

    grids_shapes = [in_shapes//{0:32, 1:16, 2:8}[l] for l in range(number_net)]

    out_data = [np.zeros((m_temp,grids_shapes[l][0],grids_shapes[l][1],len(boundingbox_mask[l]),5+number_labels),
        dtype='float32') for l in range(number_net)]

    ancors = np.expand_dims(ancors, 0)
    ancors_max_num = ancors / 2.
    ancors_min_num = -ancors_max_num

    test_mask = boundingbox_ww[..., 0]>0

    for x in range(m_temp):

        w_h = boundingbox_ww[x, test_mask[x]]
        if len(w_h)==0: continue

        w_h = np.expand_dims(w_h, -2)
        boxs_max_num = w_h / 2.
        boxs_min_num = -boxs_max_num


        insect_min = np.maximum(boxs_min_num, ancors_min_num)
        insect_max = np.minimum(boxs_max_num, ancors_max_num)
        insect_w_h = np.maximum(insect_max - insect_min, 0.)
        insect_area = insect_w_h[..., 0] * insect_w_h[..., 1]
        boxs_area = w_h[..., 0] * w_h[..., 1]
        ancor_area = ancors[..., 0] * ancors[..., 1]
        ious_bos = insect_area / (boxs_area + ancor_area - insect_area)

        best_ancor = np.argmax(ious_bos, axis=-1)

        for q, w in enumerate(best_ancor):
            for e in range(number_net):
                if w in boundingbox_mask[e]:

                    i = np.floor(ground_boundingboxs[x,q,0]*grids_shapes[e][1]).astype('int32')
                    j = np.floor(ground_boundingboxs[x,q,1]*grids_shapes[e][0]).astype('int32')

                    k = boundingbox_mask[e].index(w)
                    c = ground_boundingboxs[x,q, 4].astype('int32')
                    out_data[e][x, j, i, k, 0:4] = ground_boundingboxs[x,q, 0:4]
                    out_data[e][x, j, i, k, 4] = 1
                    out_data[e][x, j, i, k, 5+c] = 1

    return out_data


configs = tf.ConfigProto()
configs.gpu_options.allocator_type = 'BFC'
configs.gpu_options.per_process_gpu_memory_fraction = 0.7
configs.gpu_options.allow_growth = True
set_session(tf.Session(config=configs))


if __name__ == "__main__":

    xml_dir = '2007_train.txt'

    class_dir = 'model_data/voc_classes.txt'
    ancor_dir = 'model_data/yolo_anchors.txt'

    model_stat_dir = 'model_data/ep021-loss15.804-val_loss15.938.h5'

    class_names = give_total_labels(class_dir)
    anchors_box = give_total_boundingbox(ancor_dir)

    number_labels = len(class_names)
    number_box = len(anchors_box)

    loger_data_path = 'logs/'

    inputs_shapes = (416,416)


    K.clear_session()


    imgs_inputs = Input((None, None, 3))
    heigh, width = inputs_shapes


    print(' {}  {}'.format(number_box, number_labels))
    model_train = networks_yolo(imgs_inputs, number_box//3, number_labels)
    

    print(' {}.'.format(model_stat_dir))
    model_train.load_weights(model_stat_dir, by_name=True, skip_mismatch=True)
    

    labels = [Input(shape=(heigh//{0:32, 1:16, 2:8}[l], width//{0:32, 1:16, 2:8}[l], \
        number_box//3, number_labels+5)) for l in range(3)]


    loss_input = [*model_train.output, *labels]
    total_loss = Lambda(yolos_total_losss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors_box, 'num_classes': number_labels, 'ignore_thresh': 0.5})(loss_input)

    net = Model([model_train.input, *labels], total_loss)


    frzen_number_layer = 184
    for i in range(frzen_number_layer): model_train.layers[i].trainable = False
    print(' {} {} '.format(frzen_number_layer, len(model_train.layers)))

   
    logging_temp = TensorBoard(log_dir=loger_data_path)
    checkpoint_temp = ModelCheckpoint(loger_data_path + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)


    test_dataset_number = 0.1
    with open(xml_dir) as f:
        ff = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(ff)
    np.random.seed(None)
    number_test = int(len(ff)*test_dataset_number)
    number_training = len(ff) - number_test
    
    if True:
        net.compile(optimizer=Adam(lr=1e-3), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 3
        print('{}  {}  {}.'.format(number_training, number_test, batch_size))
        net.fit_generator(give_dataset_create(ff[:number_training], batch_size, imgs_inputs, anchors_box, number_labels),
                steps_per_epoch=max(1, number_training//batch_size),
                validation_data=give_dataset_create(ff[number_training:], batch_size, imgs_inputs, anchors_box, number_labels),
                validation_steps=max(1, number_test//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging_temp, checkpoint_temp, learning_rate, early_stopping])
        net.save_weights(loger_data_path + 'trained_weights_stage_1.h5')

    for i in range(frzen_number_layer): model_train.layers[i].trainable = True

    if True:
        net.compile(optimizer=Adam(lr=1e-4), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 2
        print('{} {} {}.'.format(number_training, number_test, batch_size))
        net.fit_generator(give_dataset_create(ff[:number_training], batch_size, imgs_inputs, anchors_box, number_labels),
                steps_per_epoch=max(1, number_training//batch_size),
                validation_data=give_dataset_create(ff[number_training:], batch_size, imgs_inputs, anchors_box, number_labels),
                validation_steps=max(1, number_test//batch_size),
                epochs=100,
                initial_epoch=50,
                callbacks=[logging_temp, checkpoint_temp, learning_rate, early_stopping])
        net.save_weights(loger_data_path + 'last1.h5')

