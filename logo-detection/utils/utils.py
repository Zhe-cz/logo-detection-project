from functools import reduce

from PIL import Image
import numpy as np
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2


def cobine_data(*functions):
    if functions:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), functions)
    else:
        raise ValueError('Combination of empty sequences is not supported.')


def get_boundingb_img(imgs, sizes):
    img_w, img_h = imgs.size
    img1_w, img1_h = sizes
    scales = min(img1_w / img_w, img1_h / img_h)
    num_w = int(img_w * scales)
    num_h = int(img_h * scales)

    imgs = imgs.resize((num_w, num_h), Image.BICUBIC)
    new_img = Image.new('RGB', sizes, (128, 128, 128))
    new_img.paste(imgs, ((img1_w - num_w) // 2, (img1_h - num_h) // 2))
    return new_img


def random_rand(x=0, y=1):
    return np.random.rand() * (y - x) + x


def give_rand_dataset(annotations_lines, inputs_shapes, random=True, maxs_boxes=100, J_value=.3, h_value=.1, s_value=1.5, v_value=1.5, roc_img=True):

    lines = annotations_lines.split()
    images = Image.open(lines[0])
    img_w, img_h = images.size
    img1_h, img1_w = inputs_shapes
    boxs = np.array([np.array(list(map(int, boxs.split(',')))) for boxs in lines[1:]])

    # resize image
    new_arrray = img1_w / img1_h * random_rand(1 - J_value, 1 + J_value) / random_rand(1 - J_value, 1 + J_value)
    scales_num = random_rand(.25, 2)
    if new_arrray < 1:
        num_h = int(scales_num * img1_h)
        num_w = int(num_h * new_arrray)
    else:
        num_w = int(scales_num * img1_w)
        num_h = int(num_w / new_arrray)
    images = images.resize((num_w, num_h), Image.BICUBIC)


    dd_x = int(random_rand(0, img1_w - num_w))
    dd_y = int(random_rand(0, img1_h - num_h))
    new_images = Image.new('RGB', (img1_w, img1_h), (128, 128, 128))
    new_images.paste(images, (dd_x, dd_y))
    images = new_images

    flip_cost = random_rand() < .5
    if flip_cost: images = images.transpose(Image.FLIP_LEFT_RIGHT)


    h_value = random_rand(-h_value, h_value)
    s_value = random_rand(1, s_value) if random_rand() < .5 else 1 / random_rand(1, s_value)
    v_value = random_rand(1, v_value) if random_rand() < .5 else 1 / random_rand(1, v_value)
    out = cv2.cvtColor(np.array(images, np.float32) / 255, cv2.COLOR_RGB2HSV)
    out[..., 0] += h_value * 360
    out[..., 0][out[..., 0] > 1] -= 1
    out[..., 0][out[..., 0] < 0] += 1
    out[..., 1] *= s_value
    out[..., 2] *= v_value
    out[out[:, :, 0] > 360, 0] = 360
    out[:, :, 1:][out[:, :, 1:] > 1] = 1
    out[out < 0] = 0
    new_image_data = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)  # numpy array, 0 to 1


    box_total_data = np.zeros((maxs_boxes, 5))
    if len(boxs) > 0:
        np.random.shuffle(boxs)
        boxs[:, [0, 2]] = boxs[:, [0, 2]] * num_w / img_w + dd_x
        boxs[:, [1, 3]] = boxs[:, [1, 3]] * num_h / img_h + dd_y
        if flip_cost: boxs[:, [0, 2]] = img1_w - boxs[:, [2, 0]]
        boxs[:, 0:2][boxs[:, 0:2] < 0] = 0
        boxs[:, 2][boxs[:, 2] > img1_w] = img1_w
        boxs[:, 3][boxs[:, 3] > img1_h] = img1_h
        box_w = boxs[:, 2] - boxs[:, 0]
        box_h = boxs[:, 3] - boxs[:, 1]
        box = boxs[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > maxs_boxes: box = box[:maxs_boxes]
        box_total_data[:len(box)] = box

    return new_image_data, box_total_data


def print_answer_qusestion(arg_max):
    with open("./model_data/index_word.txt", "r", encoding='utf-8') as ff:
        syssets = [ll.split(";")[1][:-1] for ll in ff.readlines()]

    return syssets[arg_max]
