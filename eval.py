import torch
from torch.autograd import Variable
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from src import SSD, DataEncoder, ListDataset, MultiBoxLoss


IMAGE_DIR = '/home/ubuntu/data/VOCdevkit/VOC2007/JPEGImages/'
LIST_FILE = 'data_utils/voc07_test_by_image.txt'
NUM_CLASSES = 20
BATCH_SIZE = 20
MODEL_PATH = 'pretrained.pth'
NMS_THRESHOLD = 0.5


def eval():
    """Trains SSD300. Saves the trained model and logs."""

    dataset = ListDataset(
        IMAGE_DIR, LIST_FILE,
        num_classes=NUM_CLASSES, train=False
    )
    iterator = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=4,
        shuffle=False, pin_memory=True, drop_last=False
    )
    data_size = len(dataset)
    decoder = dataset.data_encoder.decode

    model = SSD(NUM_CLASSES + 1).cuda()
    loss = MultiBoxLoss(NUM_CLASSES + 1).cuda()
    model.load_state_dict(torch.load(MODEL_PATH))

    n_batches = math.floor(data_size/BATCH_SIZE)
    print('number of batches:', n_batches, '\n')

    boxes, labels, confs = [], [], []
    model.eval()
    for images, loc_targets, label_targets in tqdm(iterator):

        images = Variable(images.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda(), volatile=True)
        label_targets = Variable(label_targets.cuda(), volatile=True)

        loc_preds, conf_preds = model(images)
        total_loss = loss(loc_preds, loc_targets, conf_preds, label_targets)

        for loc, conf in zip(loc_preds, conf_preds):
            output_boxes, output_labels, output_conf = decoder(loc, conf, NMS_THRESHOLD)
            boxes += [output_boxes]
            labels += [output_labels]
            confs += [output_conf]

    boxes = torch.cat(boxes).numpy()
    labels = torch.cat(labels).numpy()
    confs = torch.cat(confs).numpy()

    print('loss: {0:.2f}'.format(
        total_loss.data[0]
    ))

    _write_results(boxes, labels, confs)


def _write_results(boxes, labels, confs):
    with open('results.txt', 'w') as f:
        for b, l, c in zip(boxes, labels, confs):
            values = ('{0},{1},{2},{3},{4},{5}\n').format(*b, l, c)
            f.write(values)

            
eval()
