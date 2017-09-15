"""Convert VOC PASCAL 2007 xml annotations to a list file."""

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


VOC_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

xml_dir = '/home/dan/data/VOCdevkit/VOC2007/Annotations/'
file = open('voc07_test_by_image.txt', 'w')

for xml_name in tqdm(os.listdir(xml_dir)):
    img_name = xml_name[:-4] + '.jpg'
    file.write(img_name + ' ')
    tree = ET.parse(os.path.join(xml_dir, xml_name))
    annotations = []
    for child in tree.getroot():
        if child.tag == 'object':
            bbox = child.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            class_label = VOC_LABELS.index(child.find('name').text)
            annotations.append(
                '%s %s %s %s %s' % (xmin, ymin, xmax, ymax, class_label)
            )
    file.write('%d %s\n' % (len(annotations), ' '.join(annotations)))

file.close()
