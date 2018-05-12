import argparse
import os
import trees
import nltk
import vocabulary
import logging


logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get('PT_DATA_DIR', 'data/ptb'))
    args = parser.parse_args()
    return args


def load_data(data_dir):
    train_data = os.path.join(data_dir, '02-21.10way.clean')
    valid_data = os.path.join(data_dir, '22.auto.clean')
    test_data = os.path.join(data_dir, '23.auto.clean')

    train_trees = trees.load_trees(train_data)
    valid_trees = trees.load_trees(valid_data)
    test_trees = trees.load_trees(test_data)

    logging.warn("Converting trees...")
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

    logging.warn("Getting vocabulary...")

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

    logging.warn("Tag vocab..: %s", tag_vocab.values)
    logging.warn("Label vocab..: %s", label_vocab.values)
    logging.warn("Word vocab..: %s", word_vocab.values)

    import ipdb
    ipdb.set_trace()

    return (word_vocab, tag_vocab, label_vocab,
            train_parse, valid_parse, test_parse)


def run(args):
    data = load_data(args.data_dir)


if __name__ == '__main__':
    args = parse_args()
    run(args)
