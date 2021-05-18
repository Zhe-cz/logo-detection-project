import os
import numpy as np
import copy
import colorsys
from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model,save_model,model_from_json,clone_model
from keras.layers import Input,core,average
from PIL import Image, ImageFont, ImageDraw, FontFile, SgiImagePlugin
from nets.yolo3 import networks_yolo,yolo_test
from utils.utils import get_boundingb_img
from io import BytesIO
import cv2


def nms(bounding_boxs, thresh):

    x11 = bounding_boxs[:, 0]
    y11 = bounding_boxs[:, 1]
    x22 = bounding_boxs[:, 2]
    y22 = bounding_boxs[:, 3]


    areasq = (y22 - y11 + 1) * (x22 - x11 + 1)


    scoresq = bounding_boxs[:, 4]
    index = scoresq.argsort()[::-1]


    resq = []

    while index.size > 0:
        iq = index[0]
        resq.append(iq)

        x11q = np.maximum(x11[iq], x11[index[1:]])
        y11q = np.maximum(y11[iq], y11[index[1:]])
        x22q = np.minimum(x22[iq], x22[index[1:]])
        y22q = np.minimum(y22[iq], y22[index[1:]])

        wq = np.maximum(0, x22q - x11q + 1)
        hq = np.maximum(0, y22q - y11q + 1)

        overlapsq = wq * hq
        iousq = overlapsq / (areasq[iq] + areasq[index[1:]] - overlapsq) # index[1:]从下标1开始取到列表结束 最高分的面积加其余的面积

        idxq = np.where(iousq <= thresh)[0]
        index = index[idxq + 1]

    return resq

def mathc_img(image,Target,values_score):

    target_list = os.listdir(Target)
    for target_img_path in target_list:
        target_img = cv2.imread('./target_img/' + target_img_path)
        channels, width, height = target_img.shape[::-1]
        temp = cv2.matchTemplate(image,target_img,cv2.TM_CCOEFF_NORMED)
        res_value = values_score
        ll = np.where(temp >= res_value)
        Nims = len(ll[::-1][0])
        box = np.empty([Nims, 5])
        for i, point in enumerate(zip(*ll[::-1])):
            a = np.asarray([point[0],point[1], point[0] + width, point[1] + height, 0.7])
            box[i] = a
        new_box = nms(box, 0.7)
        for x in new_box:
            point = (ll[1][x], ll[0][x])
            cv2.rectangle(image, point, (point[0] + width, point[1] + height), (255, 0, 0), 2)
            cv2.putText(image,'logo',point,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image


class YOLO(object):
    _key_values = {
        "net_dir"        : 'logs/ep032-loss14.261-val_loss13.502.h5',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/voc_classes.txt',
        "score"             : 0.5,
        "iou"               : 0.1,
        "max_boxes"         : 100,
        "input_img_shape"  : (416, 416)
    }


    @classmethod
    def get_key_values(cls, x):
        if x in cls._key_values:
            return cls._key_values[x]
        else:
            return "name '" + x + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._key_values)
        self.labels_name = self._give_labels()
        self.boundingbox_a = self._give_boundingbox()
        self.sess = K.get_session()
        self.boundingbox, self.values, self.labels = self.creat_model()
        self.band_dict = {
        '361': 0.0,
        'adidas': 0.0,
        'anta': 0.0,
        'erke': 0.0,
        'kappa': 0.0,
        'lining': 0.0,
        'nb': 0.0,
        'nike': 0.0,
        'puma': 0.0,
        'xtep': 0.0,
    }



    def _give_labels(self):
        class_dir = os.path.expanduser(self.classes_path)
        with open(class_dir) as ff:
            labels_number = ff.readlines()
        labels_number = [x.strip() for x in labels_number]
        return labels_number


    def _give_boundingbox(self):
        ancor_dir = os.path.expanduser(self.anchors_path)
        with open(ancor_dir) as file:
            ancor_num = file.readline()
        ancor_num = [float(i) for i in ancor_num.split(',')]
        return np.asarray(ancor_num).reshape(-1, 2)


    def creat_model(self):
        net_path = os.path.expanduser(self.net_dir)
        assert net_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        

        number_anchor = len(self.boundingbox_a)
        number_class = len(self.labels_name)


        try:
            self.networks = load_model(net_path, compile=False)
        except:
            self.networks = networks_yolo(Input((None,None, 3)), number_anchor//3, number_class)
            self.networks.load_weights(self.net_dir)
        else:
            assert self.networks.layers[-1].output_shape[-1] == \
                number_anchor/len(self.networks.output) * (number_class + 5), \
                '...'

        print('{}'.format(net_path))


        color_value = [(i / len(self.labels_name), 1., 1.)
                      for i in range(len(self.labels_name))]
        self.convert_img = list(map(lambda temp: colorsys.hsv_to_rgb(*temp), color_value))
        self.convert_img = list(
            map(lambda temp: (int(temp[0] * 255), int(temp[1] * 255), int(temp[2] * 255)),
                self.convert_img))


        np.random.seed(2421)
        np.random.shuffle(self.convert_img)
        np.random.seed(None)

        self.in_img_size = K.placeholder((2, ))

        box_value, score_value, class_value = yolo_test(self.networks.output, self.boundingbox_a,
                number_class, self.in_img_size, num_boundingb = self.max_boxes,
                values = self.score, iou_vaule = self.iou)
        return box_value, score_value, class_value


    def detecter_images(self, img):
        temp_img_shape = (self.input_img_shape[1],self.input_img_shape[0])
        boundingbox_img = get_boundingb_img(img, temp_img_shape)
        img_coo= np.array(boundingbox_img, dtype='float32')
        img_coo /= 255.
        img_coo = np.expand_dims(img_coo, 0)  # Add batch dimension.

        count_boundingbox, count_values, count_labels = self.sess.run(
            [self.boundingbox, self.values, self.labels],
            feed_dict={
                self.networks.input: img_coo,
                self.in_img_size: [img.size[1], img.size[0]],
            })

        print('find {} box for {}'.format(len(count_boundingbox), 'picture'))
        font = ImageFont.truetype(font='font/simhei.ttf',
                    size=np.floor(3e-2 * img.size[1] + 0.5).astype('int32'))
        t_value = (img.size[0] + img.size[1]) // 300

        set_out_classes = count_labels
        for i in set(set_out_classes):
            set_predicted_class = self.labels_name[i]
            self.band_dict[str(set_predicted_class)] += 1

        for i, c in list(enumerate(count_labels)):
            pred_labels = self.labels_name[c]
            boundingboxs = count_boundingbox[i]
            values = count_values[i]

            tops_value, lefts_value, bottoms_value, rights_value = boundingboxs
            tops_value = tops_value - 5
            lefts_value = lefts_value - 5
            bottoms_value = bottoms_value + 5
            rights_value = rights_value + 5

            tops_value = max(0, np.floor(tops_value + 0.5).astype('int32'))
            lefts_value = max(0, np.floor(lefts_value + 0.5).astype('int32'))
            bottoms_value = min(img.size[1], np.floor(bottoms_value + 0.5).astype('int32'))
            rights_value = min(img.size[0], np.floor(rights_value + 0.5).astype('int32'))


            out_labels = '{} {:.2f}'.format(pred_labels, values)
            To_img = ImageDraw.Draw(img)
            labels_shape = To_img.textsize(out_labels, font)
            out_labels = out_labels.encode('utf-8')
            
            if tops_value - labels_shape[1] >= 0:
                txt_draw = np.array([lefts_value, tops_value - labels_shape[1]])
            else:
                txt_draw = np.array([lefts_value, tops_value + 1])

            for i in range(t_value):
                To_img.rectangle(
                    [lefts_value + i, tops_value + i, rights_value - i, bottoms_value - i],
                    outline=self.convert_img[c])
            To_img.rectangle(
                [tuple(txt_draw), tuple(txt_draw + labels_shape)],
                fill=self.convert_img[c])
            To_img.text(txt_draw, str(out_labels,'UTF-8'), fill=(0, 0, 0), font=font)
            del To_img

        r_image = np.asarray(img,dtype=np.uint8)
        target_path = './target_img'
        try:
            co_img = mathc_img(r_image, target_path, 0.7)
        except:
            co_img = r_image
        co_img = Image.fromarray(co_img)
        im = BytesIO()
        co_img.save(im, "jpeg")
        return im, img

    def close_session(self):
        self.sess.close()
