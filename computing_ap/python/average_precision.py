
IOU_THRESHOLD = 0.5


class Box:
    def __init__(self, row):
        self.img_name = row['img_name']

        # top left corner
        self.x0 = row['x']
        self.y0 = row['y']
        w, h = row['w'], row['h']

        # bottom right corner
        self.x1 = self.x0 + w
        self.y1 = self.y0 + h

        self.label = row['label']
        if len(row) > 6:
            # ground truth boxes have no confidences
            self.confidence = row['confidence']

        self.is_matched = False


def iou(box1, box2):
    w = (min(box1.x1, box2.x1) - max(box1.x0, box2.x0)) + 1
    if w > 0:
        h = (min(box1.y1, box2.y1) - max(box1.y0, box2.y0)) + 1
        if h > 0:
            intersection = w*h
            w1 = box1.x1 - box1.x0 + 1
            h1 = box1.y1 - box1.y0 + 1
            w2 = box2.x1 - box2.x0 + 1
            h2 = box2.y1 - box2.y0 + 1
            union = (w1*h1 + w2*h2) - intersection
            return float(intersection)/float(union)
    return 0.0


def match(detector_box, ground_truth_boxes):
    best_i = -1
    # ground_truth_boxes are sorted by area
    for i, gt_box in enumerate(ground_truth_boxes):
        if not gt_box.is_matched:
            computed_iou = iou(detector_box, gt_box)
            if computed_iou > IOU_THRESHOLD:
                best_i = i
                break
    return best_i


def evaluate_detector(ground_truth_boxes_by_img, detected_boxes_by_img):

    n_ground_truth_boxes = 0
    # each gt box either TP or FN

    for img in ground_truth_boxes_by_img:
        n_ground_truth_boxes += len(ground_truth_boxes_by_img[img])

    all_detected_boxes = []
    for d in detected_boxes_by_img.values():
        all_detected_boxes += d

    # sort by confidence, largest confidence is first
    all_detected_boxes.sort(key=lambda b: b.confidence, reverse=True)

    n_correctly_detected_boxes = 0
    n_detected_boxes = 0
    precision = [0.0]*len(all_detected_boxes)
    recall = [0.0]*len(all_detected_boxes)
    for k, detected_box in enumerate(all_detected_boxes):

        # each detected box either TP or FP
        n_detected_boxes += 1

        if detected_box.img_name in ground_truth_boxes_by_img:
            ground_truth_boxes = ground_truth_boxes_by_img[detected_box.img_name]
        else:
            ground_truth_boxes = []

        best_ground_truth_i = match(detected_box, ground_truth_boxes)
        is_match = best_ground_truth_i >= 0
        if is_match:
            # detected_box matched to ground truth box
            ground_truth_boxes[best_ground_truth_i].is_matched = True
            n_correctly_detected_boxes += 1  # increase number of TP

        precision[k] = float(n_correctly_detected_boxes)/float(n_detected_boxes)
        recall[k] = float(n_correctly_detected_boxes)/float(n_ground_truth_boxes)

    return compute_ap(precision, recall)


def compute_ap(precision, recall):
    prev_recall_value = 0.0
    ap = 0.0
    # recall is in increasing order
    for p, r in zip(precision, recall):
        delta = r - prev_recall_value
        ap += p*delta
        prev_recall_value = r
    return ap
