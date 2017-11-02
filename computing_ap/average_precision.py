

class Box:
    """Construct a box from a row in a pandas dataframe."""
    def __init__(self, row):
        self.image = row['image']

        # top left corner
        self.xmin = row['xmin']
        self.ymin = row['ymin']

        # bottom right corner
        self.xmax = row['xmax']
        self.ymax = row['ymax']

        self.label = row['label']
        if len(row) > 6:
            # ground truth boxes have no confidences
            self.confidence = row['confidence']

        self.is_matched = False


def evaluate_detector(ground_truth_boxes_by_img, all_detections, iou_threshold=0.5):
    """
    Arguments:
        ground_truth_boxes_by_img: a dict of lists with boxes,
            image -> list of ground truth boxes on the image.
        all_detections: a list of boxes.
        iou_threshold: a float number.
    Returns:
        a float number, average precision.
    """

    # each ground truth box is either TP or FN
    n_ground_truth_boxes = 0

    for boxes in ground_truth_boxes_by_img.values():
        n_ground_truth_boxes += len(boxes)

    # sort by confidence in decreasing order
    all_detections.sort(key=lambda box: box.confidence, reverse=True)

    n_correct_detections = 0
    n_detections = 0
    precision = [0.0]*len(all_detections)
    recall = [0.0]*len(all_detections)
    for k, detection in enumerate(all_detections):

        # each detection is either TP or FP
        n_detections += 1

        if detection.image in ground_truth_boxes_by_img:
            ground_truth_boxes = ground_truth_boxes_by_img[detection.image]
        else:
            ground_truth_boxes = []

        best_ground_truth_i = match(detection, ground_truth_boxes, iou_threshold)
        if best_ground_truth_i >= 0:
            box = ground_truth_boxes[best_ground_truth_i]
            if not box.is_matched:
                ground_truth_boxes[best_ground_truth_i].is_matched = True
                n_correct_detections += 1  # increase number of TP

        precision[k] = float(n_correct_detections)/float(n_detections)
        recall[k] = float(n_correct_detections)/float(n_ground_truth_boxes)

    return compute_ap(precision, recall)


def compute_iou(box1, box2):
    w = (min(box1.xmax, box2.xmax) - max(box1.xmin, box2.xmin)) + 1
    if w > 0:
        h = (min(box1.ymax, box2.ymax) - max(box1.ymin, box2.ymin)) + 1
        if h > 0:
            intersection = w*h
            w1 = box1.xmax - box1.xmin + 1
            h1 = box1.ymax - box1.ymin + 1
            w2 = box2.xmax - box2.xmin + 1
            h2 = box2.ymax - box2.ymin + 1
            union = (w1*h1 + w2*h2) - intersection
            return float(intersection)/float(union)
    return 0.0


def match(detection, ground_truth_boxes, iou_threshold=0.5):
    best_i = -1
    max_iou = -1.0
    for i, box in enumerate(ground_truth_boxes):
        # if not box.is_matched:
        iou = compute_iou(detection, box)
        if iou > max_iou and iou >= iou_threshold:
            best_i = i
            max_iou = iou
    return best_i


def compute_ap(precision, recall):
    previous_recall_value = 0.0
    ap = 0.0
    # recall is in increasing order
    for p, r in zip(precision, recall):
        delta = r - previous_recall_value
        ap += p*delta
        previous_recall_value = r
    return ap
