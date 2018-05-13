import math
import argparse
import os
import trees
import vocabulary
import logging
import torch
import re
import random
import functions
import tempfile
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from data import load_data, get_iterator
from loss import dist_loss, label_loss, unary_loss
from model import DistanceParser


logger = logging.getLogger("dp")
logger.setLevel(logging.INFO)


class FScore(object):
    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f})".format(
            self.recall, self.precision, self.fscore)


def get_args():
    parser = argparse.ArgumentParser(
        description='Syntactic distance based neural parser')
    parser.add_argument('--epc', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--bthsz', type=int, default=200)
    parser.add_argument('--hidsz', type=int, default=800)
    parser.add_argument('--embedsz', type=int, default=400)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--dpout', type=float, default=0.2)
    parser.add_argument('--dpoute', type=float, default=0.)
    parser.add_argument('--dpoutr', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--use_glove', action='store_true')
    parser.add_argument('--logfre', type=int, default=200)
    parser.add_argument('--devfre', type=int, default=-1)
    parser.add_argument('--cuda', action='store_true', dest='cuda')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('PT_DATA_DIR', 'data'))
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

    if nb % 10 == 0:
        print('epoch {:<3d} batch {:<4d} loss {:<.6f} rank {:<.6f} label {:<.6f} unary {:<.6f}'.format(
            epoch, nb, loss.item(), loss_dist.item(), loss_labl.item(), loss_unary.item()
        ))


def evaluate_epoch(iterator, epoch, model, vocabs):
  model.eval()
  pred_trees = []
  true_trees = []
  word_vocab, tag_vocab, label_vocab = vocabs

  # assume batch size 1
  for nb, batch in enumerate(iterator):
    words, tags, dists, labels, unarys, trees = batch
    true_tree = trees[0]
    dist_pred, label_pred, unary_pred = model(words, tags)
    dist_pred = dist_pred[0].data.cpu().numpy()
    label_pred = np.argmax(label_pred.data.cpu().numpy(), 1)
    unary_pred = np.argmax(unary_pred.data.cpu().numpy(), 1)
    label_pred = [label_vocab.value(x) for x in label_pred]
    unary_pred = [label_vocab.value(x) for x in unary_pred]
    binary_tree = functions.distance_to_tree(
        dist_pred[1:-1], label_pred[1:-1], unary_pred[1:-1],
        list(true_tree.leaves()))
    pred_tree = functions.debinarize_tree(binary_tree)
    pred_trees.append(str(pred_tree))
    true_trees.append(str(true_tree))

  temp_path = tempfile.TemporaryDirectory(prefix="evalb-")
  temp_file_path = os.path.join(temp_path.name, "pred_trees.txt")
  temp_targ_path = os.path.join(temp_path.name, "true_trees.txt")
  temp_eval_path = os.path.join(temp_path.name, "evals.txt")

  with open(temp_file_path, "w") as f:
    f.write("\n".join(pred_trees))
  with open(temp_targ_path, "w") as f:
    f.write("\n".join(true_trees))

  evalb_dir = os.path.join(args.data_dir, "EVALB")
  evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
  evalb_program_path = os.path.join(evalb_dir, "evalb")
  command = "{} -p {} {} {} > {}".format(
      evalb_program_path,
      evalb_param_path,
      temp_targ_path,
      temp_file_path,
      temp_eval_path)

  import subprocess
  subprocess.run(command, shell=True)
  fscore = FScore(math.nan, math.nan, math.nan)

  with open(temp_eval_path) as infile:
    for line in infile:
      match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
      if match:
        fscore.recall = float(match.group(1))
      match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
      if match:
        fscore.precision = float(match.group(1))
      match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
      if match:
        fscore.fscore = float(match.group(1))
        break

  temp_path.cleanup()
  success = (
    not math.isnan(fscore.fscore) or
    fscore.recall == 0.0 or
    fscore.precision == 0.0)
  return fscore


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
    train_iterator = get_iterator(train_parse, word_vocab, tag_vocab, label_vocab,
                                  args.bthsz, shuffle=True, unk_drop=False, cuda=args.cuda)
    print("Evaluating...")
    dev_iterator = get_iterator(valid_parse, word_vocab, tag_vocab, label_vocab,
                                1, shuffle=False, unk_drop=False, cuda=args.cuda)
    test_iterator = get_iterator(test_parse, word_vocab, tag_vocab, label_vocab,
                                 1, shuffle=False, unk_drop=False, cuda=args.cuda)
    train_epoch(train_iterator, epoch, model, optimizer)
    valid_fscore = evaluate_epoch(dev_iterator, epoch, model, (word_vocab, tag_vocab, label_vocab))
    test_fscore = evaluate_epoch(test_iterator, epoch, model, (word_vocab, tag_vocab, label_vocab))
    print("epoch {:d}, valid f1 {:.3f}, test f1 {:.3f}".format(
      epoch, valid_fscore.fscore, test_fscore.fscore))
    if valid_fscore.fscore is not math.nan:
      scheduler.step(valid_fscore.fscore)

if __name__ == '__main__':
  args = get_args()
  run(args)
