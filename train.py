import torch
from torch.autograd import Variable
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from src import SSD, DataEncoder, ListDataset, MultiBoxLoss


IMAGE_DIR = '~/data'
LIST_FILE = ''
NUM_CLASSES = 20
BATCH_SIZE = 20
SAVE_EVERY = 10
N_EPOCHS = 100
GRAD_CLIP = 5.0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def train():
    """Trains SSD300. Saves the trained model and logs."""

    # create a data feeder
    dataset = ListDataset(
        IMAGE_DIR, LIST_FILE,
        num_classes=NUM_CLASSES, train=True
    )
    iterator = DataLoader(
        dataset, batch_size=BATCH_SIZE, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=False
    )
    data_size = len(dataset)

    # i am adding the background class
    model = SSD(NUM_CLASSES + 1).cuda()
    loss = MultiBoxLoss(NUM_CLASSES + 1).cuda()

    for p in model.parameters():
        p.requires_grad = False

    weights = [p for n, p in model.named_parameters() if 'conv' in n and 'weight' in n]
    weights += [p for n, p in model.named_parameters() if 'multibox' in n and 'weight' in n]

    biases = [p for n, p in model.named_parameters() if 'conv' in n and 'bias' in n]
    biases += [p for n, p in model.named_parameters() if 'multibox' in n and 'bias' in n]

    for p in weights + biases:
        p.requires_grad = True

    params = [
        {'params': weights, 'weight_decay': WEIGHT_DECAY},
        {'params': biases},
    ]
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    n_batches = math.floor(data_size/BATCH_SIZE)
    print('number of batches:', n_batches, '\n')
    losses = []  # collect all losses here

    model.train()
    try:
        for epoch in range(1, N_EPOCHS + 1):
            print('epoch', epoch)
            for images, loc_targets, label_targets in tqdm(iterator):

                images = Variable(images.cuda())
                loc_targets = Variable(loc_targets.cuda())
                label_targets = Variable(label_targets.cuda())

                loc_preds, conf_preds = model(images)
                total_loss = loss(loc_preds, loc_targets, conf_preds, label_targets)

                optimizer.zero_grad()
                total_loss.backward()

                # gradient clipping by absolute value
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad = p.grad.clamp(min=-GRAD_CLIP, max=GRAD_CLIP)

                optimizer.step()

            print('loss: {0:.2f}'.format(
                total_loss.data[0]
            ))
            losses += [(epoch, total_loss.data[0])]
            if epoch % SAVE_EVERY == 0:
                print('\nsaving!\n')
                torch.save(model.state_dict(), str(epoch) + '_epoch_model_state.pytorch')

    except (KeyboardInterrupt, SystemExit):
        print(' Interruption detected, exiting the program...')

    _write_training_logs(losses)
    torch.save(model.state_dict(), 'final_model_state.pytorch')


def _write_training_logs(losses):
    with open('training_logs.txt', 'w') as f:
        column_names = 'epoch,loss\n'
        f.write(column_names)
        for i in losses:
            values = ('{0},{1:.3f}\n').format(*i)
            f.write(values)
