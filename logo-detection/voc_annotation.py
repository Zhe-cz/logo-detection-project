import xml.etree.ElementTree as ET
from os import getcwd

dataset=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

total_classes = ["361", "adidas", "anta", "erke", "kappa", "lining", "nb", "nike", "puma", "xtep","","",]

def conversion_annotation(years, img_ids, files_lst):
    inputs_files = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(years, img_ids), encoding='utf-8')
    t_num = ET.parse(inputs_files)
    root_s = t_num.getroot()

    for jet in root_s.iter('object'):
        num = 0
        if jet.find('difficult')!=None:
            num = jet.find('difficult').text
            
        temp = jet.find('name').text
        if temp not in total_classes or int(num) == 1:
            continue
        temp_id = total_classes.index(temp)
        xml_file_box = jet.find('bndbox')
        b = (int(xml_file_box.find('xmin').text), int(xml_file_box.find('ymin').text), int(xml_file_box.find('xmax').text), int(xml_file_box.find('ymax').text))
        files_lst.write(" " + ",".join([str(a) for a in b]) + ',' + str(temp_id))

wd = getcwd()

for data, img in dataset:
    img_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(data, img)).read().strip().split()
    files_lst = open('%s_%s.txt'%(data, img), 'w')
    for image_id in img_ids:
        files_lst.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, data, image_id))
        conversion_annotation(data, image_id, files_lst)
        files_lst.write('\n')
    files_lst.close()
