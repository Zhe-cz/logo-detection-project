import os
import random

val_num = 0.8
train_num = 0.75
xml_path = 'Annotations'
txt_dor = 'ImageSets\Main'
all_xml = os.listdir(txt_dor)

number = len(all_xml)
list_num = range(number)
tv_m = int(number * val_num)
tr_m = int(tv_m * train_num)
train_val = random.sample(list_num, tv_m)
train_total_num = random.sample(train_val, tr_m)


f_train_val = open('ImageSets/Main/trainval.txt', 'w')
f_test = open('ImageSets/Main/test.txt', 'w')
f_train = open('ImageSets/Main/train.txt', 'w')
f_val = open('ImageSets/Main/val.txt', 'w')

for m in list_num:
    name = all_xml[m][:-4] + '\n'
    if m in train_val:
        f_train_val.write(name)
        if m in train_total_num:
            f_train.write(name)
        else:
            f_val.write(name)
    else:
        f_test.write(name)

f_train_val.close()
f_train.close()
f_val.close()
f_test.close()