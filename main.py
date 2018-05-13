import argparse
import os
import trees
import vocabulary
import logging
from data import load_data, get_iterator
from model import DistanceParser


logger = logging.getLogger("dp")
logger.setLevel(logging.INFO)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--data_dir", type=str, default=os.environ.get('PT_DATA_DIR', 'data/ptb'))
  args = parser.parse_args()
  return args


def run(args):
  word_vocab, tag_vocab, label_vocab, \
    train_parse, valid_parse, test_parse = load_data(args.data_dir)
  model = DistanceParser(
    word_vocab.size, 400, 800, tag_vocab.size, label_vocab.size,
    dropout=0.2, dropoute=0., dropoutr=0.)
  for nb, batch in enumerate(
      get_iterator(train_parse, word_vocab, tag_vocab, label_vocab,
                   50, shuffle=True, unk_drop=False)):
    print(batch[0].shape)


if __name__ == '__main__':
  args = parse_args()
  run(args)
