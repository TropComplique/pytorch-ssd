import torch
import math
import itertools


class DataEncoder:
    """
    It transforms data before feeding the detector.
    """
    def __init__(self, num_classes=20):
        """Compute default box sizes with scale and aspect transform."""
        input_size = 300.0
        # we will divide by input_size,
        # so that bounding box coordinates are in [0,1]

        steps = [s/input_size for s in (8, 16, 32, 64, 100, 300)]
        # 8 ~ 300/38, 16 ~ 300/19, 32 ~ 300/10, ...
        # for example: one step in the first feature map corresponds
        # approximately to eight steps on the original input image,
        # but i don't completely understand why.

        scales = [s/input_size for s in (30, 60, 111, 162, 213, 264, 315)]
        # 0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05
        # (the last scale is not used directly)

        aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
        # one tuple for each scale

        # i believe we must treat steps, scales, and aspect_ratios
        # like hyperparameters, they are not exact values

        feature_map_sizes = (38, 19, 10, 5, 3, 1)
        # we know these form the network architecture

        boxes = []
        for i, fm_size in enumerate(feature_map_sizes):
            for h, w in itertools.product(range(fm_size), repeat=2):

                # center of a default box
                cx = (w + 0.5)*steps[i]
                cy = (h + 0.5)*steps[i]

                s = scales[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(scales[i] * scales[i + 1])
                boxes.append((cx, cy, s, s))

                s = scales[i]
                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s*math.sqrt(ar), s/math.sqrt(ar)))
                    boxes.append((cx, cy, s/math.sqrt(ar), s*math.sqrt(ar)))

        # there are 8732 default boxes overall,
        # 4*(38**2) + 6*(19**2) + 6*(10**2) + 6*(5**2) + 4*(3**2) + 4*(1**2) = 8732
        self.default_boxes = torch.FloatTensor(boxes)

        self.variances = [0.1, 0.2]
        # you can read about variances here:
        # github.com/rykov8/ssd_keras/issues/53
        # github.com/weiliu89/caffe/issues/155

        self.num_classes = num_classes

    def encode(self, boxes, classes, threshold=0.5):
        """Encode ground truth bounding boxes and class labels to
        format that is used for training of SSD.
        Note: it takes as input boxes and classes for one image, not batch.

        Arguments:
            boxes: a float tensor of shape [#obj, 4],
                object bounding boxes (x_min, y_min, x_max, y_max) of an image.
            classes: a long tensor of shape [#obj],
                object class labels of an image.
            threshold: a float number, Jaccard index (IoU) threshold.

        Returns:
            loc: a float tensor of shape [8732, 4],
                transformations of default boxes.
            label: a long tensor of shape [8732], class labels.
        """
        default_boxes = self.default_boxes
        variances = self.variances

        # if we have a default box (cx, cy, w, h) then
        # x0 = cx - w/2, y0 = cy - h/2
        # x1 = cx + w/2, y1 = cy + h/2

        if boxes.numel() == 0:
            return torch.zeros(8732, 4), torch.zeros(8732)

        # compute IoU between each ground truth box and all default boxes
        iou = compute_iou(
            boxes,
            torch.cat([default_boxes[:, :2] - 0.5*default_boxes[:, 2:],
                       default_boxes[:, :2] + 0.5*default_boxes[:, 2:]], 1)
        )  # [#obj, 8732], '#obj' is boxes.size(0), number of ground truth boxes

        # assign a ground truth box to each default box
        iou, max_idx = iou.max(0)
        max_idx = max_idx.squeeze()  # [8732]
        iou = iou.squeeze()  # [8732]
        boxes = boxes[max_idx]  # [8732, 4]

        # in other words: for each default box we find
        # a ground truth box that has largest IoU

        # but it could happen that some ground truth box has
        # no default boxes assigned to it

        # compute distances between ground truth boxes and default boxes
        dist = 0.5*(boxes[:, :2] + boxes[:, 2:]) - default_boxes[:, :2]  # [8732, 2]
        dist /= default_boxes[:, 2:]*variances[0]

        # compute ratio of ground truth boxes' lengths and default boxes' lengths
        ratio = (boxes[:, 2:] - boxes[:, :2])/default_boxes[:, 2:]  # [8732, 2]
        ratio = torch.log(ratio)/variances[1]

        loc = torch.cat([dist, ratio], 1)  # [8732, 4]
        # 'loc' here is not locations but transformations
        # from default boxes to ground truth boxes

        label = 1 + classes[max_idx]  # [8732]
        # background class is 0

        # background is where overlap is too small
        label[iou < threshold] = 0
        # so now transformations (elements of 'loc')
        # where iou < threshold are kinda meaningless,
        # because in bounding box regression we
        # are using only matched boxes (boxes with iou >= threshold)

        # so, for each default box we assigned an object with its class,
        # some default boxes are background, it's where overlap with
        # any ground truth box is small
        return loc, label

    def decode(self, loc, conf, nms_threshold=0.5):
        """Interpret output of SSD.

        Transforms predicted loc/conf back to real
        bounding box locations and class labels.
        Note: it takes as input predictions for one image, not batch.

        Arguments:
            loc: a float tensor of shape [8732, 4].
            conf: a float tensor of shape [8732, num_classes + 1].

        Returns:
            output_boxes: a float tensor of shape [#obj, 4],
                in the form (x_min, y_min, x_max, y_max).
            output_labels: a long tensor of shape [#obj].
            output_conf: a float tensor of shape [#obj].
        """
        default_boxes = self.default_boxes
        variances = self.variances

        # for each default box find label with largest confidence
        max_conf, labels = conf.max(1)  # [8732], [8732]

        # find default boxes where maximum confidence is for non background
        ids = labels.nonzero().squeeze()  # [#boxes]
        if ids.numel() == 0:
            return torch.FloatTensor([]), torch.LongTensor([]), torch.FloatTensor([])
        labels = labels[ids]
        max_conf = max_conf[ids]
        loc = loc[ids]
        default_boxes = default_boxes[ids]
        # '#boxes' is number of default boxes with predicted non background

        # transform default bounding boxes to predicted bounding boxes
        cxcy = default_boxes[:, :2] + default_boxes[:, 2:]*(loc[:, :2]*variances[0])
        wh = default_boxes[:, 2:]*(loc[:, 2:]*variances[1]).exp()
        boxes = torch.cat([cxcy - 0.5*wh, cxcy + 0.5*wh], 1)  # [#boxes, 4]
        boxes = boxes.clamp(min=0.0, max=1.0)

        output_boxes, output_labels, output_conf = [], [], []
        # do nms for each label
        for label in range(1, self.num_classes + 1):
            mask = (labels == label).nonzero().squeeze()
            if mask.numel() == 0:
                continue
            keep = nms(boxes[mask], max_conf[mask], nms_threshold)
            output_boxes += [boxes[mask][keep]]
            output_labels += [labels[mask][keep]]
            output_conf += [max_conf[mask][keep]]

        output_boxes = torch.cat(output_boxes, 0)
        output_labels = torch.cat(output_labels, 0)
        output_conf = torch.cat(output_conf, 0)

        # transform labels to range from 0 to (num_classes - 1)
        output_labels -= 1

        return output_boxes, output_labels, output_conf


def compute_iou(boxes1, boxes2):
    """Compute IoU of two sets of boxes, each box is [x1, y1, x2, y2].

    Arguments:
        boxes1: a float tensor of shape [n, 4].
        boxes2: a float tensor of shape [m, 4].

    Returns:
        a float tensor of shape [n, m].
    """
    n = boxes1.size(0)
    m = boxes2.size(0)

    # left top
    lt = torch.max(
        boxes1[:, :2].unsqueeze(1).expand(n, m, 2),
        boxes2[:, :2].unsqueeze(0).expand(n, m, 2),
    )
    # [n, 2] -> [n, 1, 2] -> [n, m, 2]
    # [m, 2] -> [1, m, 2] -> [n, m, 2]

    # right bottom
    rb = torch.min(
        boxes1[:, 2:].unsqueeze(1).expand(n, m, 2),
        boxes2[:, 2:].unsqueeze(0).expand(n, m, 2),
    )

    # width height
    wh = rb - lt  # [n, m, 2]
    wh[wh < 0.0] = 0.0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [n, m]

    area1 = (boxes1[:, 2] - boxes1[:, 0])*(boxes1[:, 3] - boxes1[:, 1])  # [n]
    area2 = (boxes2[:, 2] - boxes2[:, 0])*(boxes2[:, 3] - boxes2[:, 1])  # [m]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [n] -> [n, 1] -> [n, m]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [m] -> [1, m] -> [n, m]

    iou = inter/(area1 + area2 - inter)
    return iou


def nms(bboxes, scores, threshold=0.5, mode='union'):
    """Non maximum suppression.

    Arguments:
        bboxes: a float tensor of shape [n, 4].
        scores: a float tensor of shape [n].
        threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        a long tensor of shape [m], selected indices.
    """

    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]

    areas = (x2 - x1)*(y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:

        # element with the biggest score
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        # find intersections of the biggest score box
        # and the rest of the boxes
        ix1 = x1[order[1:]].clamp(min=x1[i])
        iy1 = y1[order[1:]].clamp(min=y1[i])
        ix2 = x2[order[1:]].clamp(max=x2[i])
        iy2 = y2[order[1:]].clamp(max=y2[i])

        w = (ix2 - ix1).clamp(min=0.0)
        h = (iy2 - iy1).clamp(min=0.0)
        inter = w*h

        if mode == 'union':
            overlap = inter/(areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            overlap = inter/areas[order[1:]].clamp(max=areas[i])

        ids = (overlap <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.LongTensor(keep)
