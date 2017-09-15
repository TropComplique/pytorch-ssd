import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from encoder import DataEncoder
from PIL import Image
import cv2


def random_flip(self, img, boxes):
    '''Randomly flip the image and adjust the bbox locations.

    For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
    (w-xmax, ymin, w-xmin, ymax).

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bbox locations, sized [#obj, 4].

    Returns:
      img: (PIL.Image) randomly flipped image.
      boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
    return img, boxes


def random_crop(self, img, boxes, labels):
    '''Randomly crop the image and adjust the bbox locations.

    For more details, see 'Chapter2.2: Data augmentation' of the paper.

    Args:
      img: (PIL.Image) image.
      boxes: (tensor) bbox locations, sized [#obj, 4].
      labels: (tensor) bbox labels, sized [#obj,].

    Returns:
      img: (PIL.Image) cropped image.
      selected_boxes: (tensor) selected bbox locations.
      labels: (tensor) selected bbox labels.
    '''
    imw, imh = img.size
    while True:
        min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
        if min_iou is None:
            return img, boxes, labels

        for _ in range(100):
            w = random.randrange(int(0.1*imw), imw)
            h = random.randrange(int(0.1*imh), imh)

            if h > 2*w or w > 2*h:
                continue

            x = random.randrange(imw - w)
            y = random.randrange(imh - h)
            roi = torch.Tensor([[x, y, x+w, y+h]])

            center = (boxes[:, :2] + boxes[:, 2:]) / 2  # [N,2]
            roi2 = roi.expand(len(center), 4)  # [N,4]
            mask = (center > roi2[:, :2]) & (center < roi2[:, 2:])  # [N,2]
            mask = mask[:, 0] & mask[:, 1]  # [N,]
            if not mask.any():
                continue

            selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

            iou = self.data_encoder.iou(selected_boxes, roi)
            if iou.min() < min_iou:
                continue

            img = img.crop((x, y, x+w, y+h))
            selected_boxes[:, 0].add_(-x).clamp_(min=0, max=w)
            selected_boxes[:, 1].add_(-y).clamp_(min=0, max=h)
            selected_boxes[:, 2].add_(-x).clamp_(min=0, max=w)
            selected_boxes[:, 3].add_(-y).clamp_(min=0, max=h)
            return img, selected_boxes, labels[mask]
