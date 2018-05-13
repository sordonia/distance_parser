import os
import trees
import logging
import random
import numpy as np
import vocabulary
import torch
from functions import *


logger = logging.getLogger("dp")
logger.setLevel(logging.INFO)
# train_data = os.path.join(data_dir, '22.auto.clean')


def load_data(data_dir):
    train_data = os.path.join(data_dir, '02-21.10way.clean')
    # train_data = os.path.join(data_dir, '22.auto.clean')
    valid_data = os.path.join(data_dir, '22.auto.clean')
    test_data = os.path.join(data_dir, '23.auto.clean')

    print("Reading trees...")
    train_trees = trees.load_trees(train_data)
    valid_trees = trees.load_trees(valid_data)
    test_trees = trees.load_trees(test_data)

    print("Converting trees...")
    train_parse = [tree.convert() for tree in train_trees]
    valid_parse = [tree.convert() for tree in valid_trees]
    test_parse = [tree.convert() for tree in test_trees]

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(vocabulary.PAD)
    tag_vocab.index(vocabulary.START)
    tag_vocab.index(vocabulary.STOP)
    tag_vocab.index(vocabulary.UNK)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(vocabulary.PAD)
    word_vocab.index(vocabulary.START)
    word_vocab.index(vocabulary.STOP)
    word_vocab.index(vocabulary.UNK)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(vocabulary.PAD)
    label_vocab.index(())

    print("Getting vocabulary...")
    for tree in train_parse:
      nodes = [tree]
      while nodes:
        node = nodes.pop()
        if isinstance(node, trees.InternalParseNode):
          label_vocab.index(node.label)
          nodes.extend(reversed(node.children))
        else:
          tag_vocab.index(node.tag)
          word_vocab.index(node.word)

    label_vocab.freeze()
    word_vocab.freeze()
    tag_vocab.freeze()

    print("Tag vocab: ", tag_vocab.size)
    print("Label vocab: ", label_vocab.size)
    print("Word vocab: ", word_vocab.size)

    return (word_vocab, tag_vocab, label_vocab,
            train_parse, valid_parse, test_parse)


def _pad(batch, type='int64', cuda=False):
  max_len = max(len(v) for v in batch)
  pad_batch = np.zeros((len(batch), max_len))
  for i, row in enumerate(batch):
    pad_batch[i, :len(row)] = row
  pad_batch = torch.from_numpy(pad_batch.astype(type))
  if cuda:
    pad_batch = pad_batch.cuda()
  return pad_batch


def _index(t, v, token=vocabulary.UNK):
  return v.index(t) if v.count(t) > 0 else v.index(token)


def get_iterator(trees, word_vocab, tag_vocab, label_vocab,
                 batch_size, shuffle=True, unk_drop=True, cuda=False):

  idxs = list(range(len(trees)))
  random.shuffle(idxs)

  for idx in range(0, len(idxs), batch_size):
    ridxs = idxs[idx:idx + batch_size]
    trees_ = [trees[i] for i in ridxs]
    stats_ = [tree_to_distance(binarize_tree(tree)) for tree in trees_]
    # sanity check
    for tree, stat in zip(trees_, stats_):
      assert str(tree) == str(debinarize_tree(distance_to_tree(
          stat[0], stat[1], stat[2], list(tree.leaves()))))

    dists_, labels_, unarys_, _ = zip(*stats_)
    sents_ = [[(l.tag, l.word) for l in tree.leaves()] for tree in trees_]
    words_, tags_ = [], []

    for sent in sents_:
      words__ = []
      tags__ = []
      for (tag, word) in [('<S>', '<S>')] + sent + [('</S>', '</S>')]:
        words__.append(_index(word, word_vocab))
        tags__.append(_index(tag, tag_vocab))
      words_.append(words__)
      tags_.append(tags__)

    # default value only for testing (not used anyways)
    labels_ = [[0] + [_index(label, label_vocab, token=()) for label in labels] + [0] for labels in labels_]
    unarys_ = [[0] + [_index(label, label_vocab, token=()) for label in labels] + [0] for labels in unarys_]
    dists_ = [[0] + dist_ + [0] for dist_ in dists_]
    yield (_pad(words_, cuda=cuda), _pad(tags_, cuda=cuda), _pad(dists_, cuda=cuda),
           _pad(labels_, cuda=cuda), _pad(unarys_, cuda=cuda), trees_)

