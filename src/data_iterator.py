import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from .encoder import compute_iou, DataEncoder


"""Load image/classes/bounding boxes from an annotation file.

The annotation file is organized as:
image_name #obj xmin ymin xmax ymax class_index xmin2 ymin2 xmax2 ymax2 class_index2 ...
"""


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, num_classes=20, train=False):
        """
        Arguments:
            root: a string, directory with images.
            list_file: a string, path to an index file.
            num_classes: an integer, number of classes (without background).
            train: boolean, train or test.
        """
        self.img_size = 300
        self.root = root
        self.train = train

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder(num_classes)
        self.mean = np.array([104.0, 117.0, 123.0], 'float32')

        with open(list_file) as file:
            lines = file.readlines()
            self.num_images = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])

            num_objects = int(splited[1])
            boxes, labels = [], []
            for i in range(num_objects):
                x_min = splited[2 + 5*i]
                y_min = splited[3 + 5*i]
                x_max = splited[4 + 5*i]
                y_max = splited[5 + 5*i]
                c = splited[6 + 5*i]
                boxes.append([
                    float(x_min), float(y_min),
                    float(x_max), float(y_max)
                ])
                labels.append(int(c))

            self.boxes.append(torch.FloatTensor(boxes))
            self.labels.append(torch.LongTensor(labels))

    def __getitem__(self, idx):
        """Load an image, then encode its bounding box locations and class labels.

        Arguments:
            idx: an integer in the range 0..(n_images - 1).

        Returns:
            img: a float tensor of shape [300, 300].
            loc_target: a float tensor of shape [8732, 4].
            label_target: a long tensor of shape [8732].
        """

        # load image and bounding boxes
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root, fname), cv2.IMREAD_COLOR)
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if boxes.numel() != 0:
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
        ).astype('float32')
        img = img - self.mean
        img = img.transpose((2, 0, 1))
        img = torch.FloatTensor(img)

        loc_target, label_target = self.data_encoder.encode(boxes, labels)

        if self.train:
            return img, loc_target, label_target
        else:
            # when evaluating we also want to know image filename
            return torch.LongTensor([idx]), img, loc_target, label_target

    def __len__(self):
        return self.num_images


def random_flip(img, boxes):
    """Randomly flip an image and its bounding boxes.

    For a bounding box (xmin, ymin, xmax, ymax), the flipped
    bounding box is: (w - xmax, ymin, w - xmin, ymax).

    Arguments:
        img: a numpy byte array, image.
        boxes: a float tensor with shape [#obj, 4].

    Returns:
        img: a numpy byte array, image.
        boxes: a float tensor with shape [#obj, 4].
    """
    if np.random.uniform() < 0.5:
        img = cv2.flip(img, 1)
        _, width, _ = img.shape
        xmin = width - boxes[:, 2]
        xmax = width - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
    return img, boxes


def random_crop(img, boxes, labels):
    """Randomly crop an image and
    adjust the bounding box locations.

    Arguments:
        img: a numpy byte array, image.
        boxes: a float tensor with shape [#obj, 4].
        labels: a long tensor with shape [#obj, 4].

    Returns:
        img: a numpy byte array, image.
        selected_boxes: a float tensor with shape [#obj', 4].
        labels: a long tensor with shape [#obj', 4].
            where #obj' <= #obj.
    """
    height, width, _ = img.shape
    while True:

        # minimal overlap between a cropped window
        # and ground truth bounding boxes
        min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
        if min_iou is None:
            return img, boxes, labels

        for _ in range(100):
            # width and height of a crop
            w = np.random.randint(int(0.1*width), width)
            h = np.random.randint(int(0.1*height), height)

            # ratio must be less than 2
            if h > 2*w or w > 2*h:
                continue

            # left top corner of the crop
            x = np.random.randint(width - w)
            y = np.random.randint(height - h)
            # the crop
            roi = torch.FloatTensor([[x, y, x + w, y + h]])

            center = 0.5*(boxes[:, :2] + boxes[:, 2:])  # [#obj, 2]
            roi2 = roi.expand_as(center)  # [#obj, 4]

            # centers inside the crop
            mask = (center > roi2[:, :2]) & (center < roi2[:, 2:])  # [#obj, 2]
            mask = mask[:, 0] & mask[:, 1]  # [#obj]
            if not mask.any():
                continue

            selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))
            iou = compute_iou(selected_boxes, roi)
            if iou.min() < min_iou:
                continue

            img = img[y:(y + h), x:(x + w)]
            selected_boxes[:, 0].add_(-x).clamp_(min=0, max=w)
            selected_boxes[:, 1].add_(-y).clamp_(min=0, max=h)
            selected_boxes[:, 2].add_(-x).clamp_(min=0, max=w)
            selected_boxes[:, 3].add_(-y).clamp_(min=0, max=h)
            labels = labels[mask]
            return img, selected_boxes, labels
