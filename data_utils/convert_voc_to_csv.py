"""Convert VOC PASCAL 2007 xml annotations to CSV format."""

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

xml_dir = '/home/ubuntu/data/VOCdevkit/VOC2007/Annotations/'
file = open('voc07_test.csv', 'w')
file.write('image,xmin,ymin,xmax,ymax,label\n')

for xml_name in tqdm(os.listdir(xml_dir)):
    img_name = xml_name[:-4] + '.jpg'
    tree = ET.parse(os.path.join(xml_dir, xml_name))
    for child in tree.getroot():
        if child.tag == 'object':
            is_difficult = int(child.find('difficult').text) == 1
            if is_difficult:
                continue
            bbox = child.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            class_label = child.find('name').text
            file.write(
                '%s,%s,%s,%s,%s,%s\n' % (img_name, xmin, ymin, xmax, ymax, class_label)
            )

file.close()
