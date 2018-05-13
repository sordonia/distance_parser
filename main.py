import argparse
import os
import trees
import vocabulary
import logging
import torch
import random
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import load_data, get_iterator
from loss import dist_loss, label_loss, unary_loss
from model import DistanceParser


logger = logging.getLogger("dp")
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(
        description='Syntactic distance based neural parser')
    parser.add_argument('--epc', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--bthsz', type=int, default=20)
    parser.add_argument('--hidsz', type=int, default=800)
    parser.add_argument('--embedsz', type=int, default=400)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--dpout', type=float, default=0.3)
    parser.add_argument('--dpoute', type=float, default=0.)
    parser.add_argument('--dpoutr', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--use_glove', action='store_true')
    parser.add_argument('--logfre', type=int, default=200)
    parser.add_argument('--devfre', type=int, default=-1)
    parser.add_argument('--cuda', action='store_true', dest='cuda')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('PT_DATA_DIR', 'ptb'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('PT_OUTPUT_DIR', 'results'))
    args = parser.parse_args()
    # set seed and return args
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.random.manual_seed(args.seed)
    return args


def train_epoch(iterator, epoch, model, optimizer):
  model.train()
  
  for nb, batch in enumerate(iterator):
    words, tags, dists, labels, unarys, trees = batch
    dist_pred, label_pred, unary_pred = model(words, tags)
    loss_dist = dist_loss(dist_pred, dists)
    loss_labl = label_loss(label_pred, labels.view(-1))
    loss_unary = unary_loss(unary_pred, unarys.view(-1))
    optimizer.zero_grad()

    loss = loss_dist + loss_labl + loss_unary
    loss.backward()
    optimizer.step()
  
    print('epoch {:<3d} batch {:<4d} loss {:<.6f} rank {:<.6f} arc {:<.6f} tag {:<.6f}'.format(
        epoch, nb, loss.item(), loss_dist.item(), loss_labl.item(), loss_unary.item()
    ))


def run(args):
  word_vocab, tag_vocab, label_vocab, \
  train_parse, valid_parse, test_parse = load_data(args.data_dir)
  model = DistanceParser(
    word_vocab.size, 400, 800, tag_vocab.size, label_vocab.size,
    dropout=0.2, dropoute=0., dropoutr=0.)
  if args.cuda:
    model = model.cuda()
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.999),
                               weight_decay=args.weight_decay)
  scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, min_lr=0.000001)
  for epoch in range(args.epc):
    iterator = get_iterator(train_parse, word_vocab, tag_vocab, label_vocab,
                            args.batch_size, shuffle=True, unk_drop=False, cuda=args.cuda)
    train_epoch(iterator, epoch, model, optimizer)
    
  

if __name__ == '__main__':
  args = get_args()
  run(args)
