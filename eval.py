import torch
from torch.autograd import Variable
import math
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from src import SSD, DataEncoder, ListDataset, MultiBoxLoss


IMAGE_DIR = '/home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages/'
LIST_FILE = 'data_utils/voc07_test_by_image.txt'
NUM_CLASSES = 20
BATCH_SIZE = 128
MODEL_PATH = 'pretrained.pth'
NMS_THRESHOLD = 0.5
VOC_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]


def eval():
    """Trains SSD300. Saves the trained model and logs."""

    dataset = ListDataset(
        IMAGE_DIR, LIST_FILE,
        num_classes=NUM_CLASSES,
        train=False
    )
    iterator = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=4,
        shuffle=False, pin_memory=True, drop_last=False
    )
    data_size = len(dataset)

    # for interpreting an output of the network
    decoder = dataset.data_encoder.decode
    # it does NMS and outputs final predictions

    # filenames of images in the dataset
    image_names = dataset.fnames

    model = SSD(NUM_CLASSES + 1).cuda()
    loss = MultiBoxLoss(NUM_CLASSES + 1).cuda()
    model.load_state_dict(torch.load(MODEL_PATH))

    n_batches = math.ceil(data_size/BATCH_SIZE)
    print('number of batches:', n_batches, '\n')

    filenames, boxes, labels, confs = [], [], [], []
    model.eval()
    for ids, images, loc_targets, label_targets in tqdm(iterator):

        images = Variable(images.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda(), volatile=True)
        label_targets = Variable(label_targets.cuda(), volatile=True)

        loc_preds, conf_preds = model(images)
        total_loss = loss(loc_preds, loc_targets, conf_preds, label_targets)

        # compute softmax
        logits = conf_preds  # [BATCH_SIZE, 8732, NUM_CLASSES + 1]
        exp = logits.exp()
        conf_preds = exp/exp.sum(2, keepdim=True)

        loc_preds = loc_preds.data.cpu()
        conf_preds = conf_preds.data.cpu()

        # for each image in the batch get final predictions
        for i, loc, conf in zip(ids, loc_preds, conf_preds):
            output_boxes, output_labels, output_conf = decoder(loc, conf, NMS_THRESHOLD)
            if output_boxes.numel() != 0:

                # scale bounding boxes coordinates to absolute
                # values x_min, y_min, x_max, y_max
                name = image_names[i]
                h, w = cv2.imread(os.path.join(dataset.root, fname), cv2.IMREAD_COLOR).shape[:2]
                output_boxes /= torch.FloatTensor([w, h, w, h]).expand_as(output_boxes)

                filenames += [name]*output_boxes.size(0)
                boxes += [output_boxes]
                labels += [output_labels]
                confs += [output_conf]

    boxes = torch.cat(boxes).numpy()
    labels = torch.cat(labels).numpy()
    confs = torch.cat(confs).numpy()

    print('loss: {0:.2f}'.format(
        total_loss.data[0]
    ))

    _write_results(filenames, boxes, labels, confs)


def _write_results(filenames, boxes, labels, confs):
    with open('results.txt', 'w') as f:
        for f, b, l, c in zip(filenames, boxes, labels, confs):
            values = (f + ',{0},{1},{2},{3},{4},{5}\n').format(*b, VOC_LABELS[l], c)
            f.write(values)


eval()
