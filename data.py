import os
import trees
import logging
import random
import numpy as np
import vocabulary
import torch
from functions import binarize_tree, tree_to_distance


logger = logging.getLogger("dp")
logger.setLevel(logging.INFO)


def load_data(data_dir):
    # train_data = os.path.join(data_dir, '02-21.10way.clean')
    train_data = os.path.join(data_dir, '22.auto.clean')
    valid_data = os.path.join(data_dir, '22.auto.clean')
    test_data = os.path.join(data_dir, '23.auto.clean')

    train_trees = trees.load_trees(train_data)
    valid_trees = trees.load_trees(valid_data)
    test_trees = trees.load_trees(test_data)

    logger.info("Converting trees...")
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
    label_vocab.index(())

    logger.info("Getting vocabulary...")

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

    logger.info("Tag vocab..: %s", tag_vocab.values)
    logger.info("Label vocab..: %s", label_vocab.values)
    logger.info("Word vocab..: %s", word_vocab.values[:100])

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


def get_iterator(trees, word_vocab, label_vocab, tag_vocab,
                 batch_size, shuffle=True, unk_drop=True, cuda=False):
  
  idxs = list(range(len(trees)))
  random.shuffle(idxs)

  for idx in range(0, len(idxs), batch_size):
    ridxs = idxs[idx:idx + batch_size]
    trees_ = [trees[i] for i in ridxs]
    stats_ = [tree_to_distance(binarize_tree(tree)) for tree in trees_]
    
    dists_, labels_, unarys_, _ = zip(*stats_)
    tags_ = [[tag_vocab.index(leaf.tag) for leaf in tree.leaves()] for tree in trees_]
    words_ = [[word_vocab.index(leaf.word) for leaf in tree.leaves()] for tree in trees_]
    labels_ = [[label_vocab.index(label) for label in labels] for labels in labels_]
    unarys_ = [[label_vocab.index(label) for label in labels] for labels in unarys_]
    yield (_pad(words_, cuda=cuda), _pad(tags_, cuda=cuda), _pad(dists_, cuda=cuda),
           _pad(labels_, cuda=cuda), _pad(unarys_, cuda=cuda), trees_)

