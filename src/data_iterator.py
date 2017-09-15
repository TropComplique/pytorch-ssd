import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from encoder import DataEncoder
from PIL import Image
import cv2


"""Load image/class/bounding box from an annotation file.

The annotation file is organized as:
image_name #obj xmin ymin xmax ymax class_index xmin2 ymin2 xmax2 ymax2 class_index2 ...
"""


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, transform, train=False):
        """
        Arguments:
            root: a string, directory with images.
            list_file: a string, path to an index file.
            transforms: a list of image transforms.
            train: boolean, train or test.
        """
        self.img_size = 300
        self.root = root
        self.train = train
        self.transforms = transforms

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder()

        with open(list_file) as file:
            lines = file.readlines()
            self.num_images = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])

            num_objects = int(splited[1])
            box, label = [], []
            for i in range(num_objects):
                x_min = splited[2 + 5*i]
                y_min = splited[3 + 5*i]
                x_max = splited[4 + 5*i]
                y_max = splited[5 + 5*i]
                c = splited[6 + 5*i]
                box.append([
                    float(x_min), float(y_min),
                    float(x_max), float(y_max)
                ])
                label.append(int(c))

            self.boxes.append(torch.FloatTensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        """Load an image, then encode its bounding box locations and class labels.

        Arguments:
            idx: an integer in the range 0..(n_ground_truth_bounding_boxes - 1).

        Returns:
            img: a float tensor of shape [300, 300].
            loc_target: a float tensor of shape [8732, 4].
            label_target: a long tensor of shape [8732].
        """

        # load image and bounding boxes
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname), cv2.IMREAD_COLOR)
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        # data augmentation while training
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes, labels = random_crop(img, boxes, labels)

        # scale bounding box coordinates to [0,1]
        h, w = img.shape[:2]
        boxes /= torch.FloatTensor([w, h, w, h]).expand_as(boxes)

        img = cv2.resize(
            img, ((self.img_size, self.img_size)),
            interpolation=cv2.INTER_LINEAR
        )
        # do something here!!!
        img = self.transform(img)

        loc_target, label_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, label_target

    def __len__(self):
        return self.num_images
