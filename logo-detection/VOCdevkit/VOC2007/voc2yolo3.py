import os
import random 
 
xml_file_dir='D:\logo-detection-v2\VOCdevkit\VOC2007\Annotations'
save_Base_dir=r"./VOCdevkit/VOC2007/ImageSets/Main/"
 
train_val_const=1.0
train_num_const=0.9

temps_xml_list = os.listdir(xml_file_dir)
totals_xmls_number = []
for xmls in temps_xml_list:
    if xmls.endswith(".xml"):
        totals_xmls_number.append(xmls)

numbers = len(totals_xmls_number)
list_num = range(numbers)
true_v = int(numbers*train_val_const)
true_recal = int(true_v*train_num_const)
train_val_sample = random.sample(list_num,true_v)
train_sample = random.sample(train_val_sample,true_recal)
 
print("total size:", true_v)
print("true spe:", true_recal)
train_val_sample_file = open('D:/logo-detection-v2/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
test_sample_file = open('D:/logo-detection-v2/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
train_sample_file = open('D:/logo-detection-v2/VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
val_sample_file = open('D:/logo-detection-v2/VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')
 
for i  in list_num:
    name=totals_xmls_number[i][:-4]+'\n'
    if i in train_val_sample:
        train_val_sample_file.write(name)
        if i in train_sample:
            train_sample_file.write(name)
        else:  
            val_sample_file.write(name)
    else:  
        test_sample_file.write(name)
  
train_val_sample_file.close()
train_sample_file.close()
val_sample_file.close()
test_sample_file .close()
