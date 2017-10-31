import torch
from torch.autograd import Variable
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from src import SSD, DataEncoder, ListDataset, MultiBoxLoss


IMAGE_DIR = '~/data'
LIST_FILE = ''
NUM_CLASSES = 20
BATCH_SIZE = 20
MODEL_PATH = ''


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
            boxes += output_boxes
            labels += output_labels
            confs += output_conf

    boxes = torch.cat(boxes)
    labels = torch.cat(labels)
    confs = torch.cat(confs)

    print('loss: {0:.2f}'.format(
        total_loss.data[0]
    ))

    _write_training_logs(losses)


def _write_training_logs(losses):
    with open('training_logs.txt', 'w') as f:
        column_names = 'epoch,loss\n'
        f.write(column_names)
        for i in losses:
            values = ('{0},{1:.3f}\n').format(*i)
            f.write(values)
