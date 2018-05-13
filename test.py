import os
import trees
import numpy as np
import vocabulary

test_data = os.path.join('data/ptb', '23.auto.clean')
test_trees = trees.load_trees(test_data)
test_parse = [tree.convert() for tree in test_trees]

tree = test_parse[0]
sent = [(t.tag, t.word) for t in tree.leaves()]


def binarize_tree(tree):
  if isinstance(tree, trees.LeafParseNode):
    return tree
  else:
    if len(tree.children) == 1:
      return tree
    elif len(tree.children) == 2:
      left_child = binarize_tree(tree.children[0])
      right_child = binarize_tree(tree.children[1])
    else:
      left_child = binarize_tree(tree.children[0])
      right_child = binarize_tree(trees.InternalParseNode((), tree.children[1:]))
    return trees.InternalParseNode(tree.label, [left_child, right_child])


def debinarize_tree(tree):
  if isinstance(tree, trees.LeafParseNode):
    return [tree]
  children = []
  for child in tree.children:
    children.extend(debinarize_tree(child))
  if tree.label:
    return [trees.InternalParseNode(tree.label, children)]
  return children


def tree_to_distance(root):
  if isinstance(root, trees.InternalParseNode) and len(root.children) == 2:
    d_l, c_l, t_l, h_l = tree_to_distance(root.children[0])
    d_r, c_r, t_r, h_r = tree_to_distance(root.children[1])
    h = max(h_l, h_r) + 1
    d = d_l + [h] + d_r
    c = c_l + [root.label] + c_r
    t = t_l + t_r
  else:
    # unary chain
    d = []
    c = []
    h = 0
    if isinstance(root, trees.InternalParseNode):
      assert len(root.children) == 1
      t = [root.label]
    else:
      # leaf
      assert isinstance(root, trees.LeafParseNode)
      t = [vocabulary.TAG_UNK]
  return d, c, t, h


def distance_to_tree(dist, cons, tags, leaves):
  if dist == []:
    tree = leaves[0]
    if tags[0] != vocabulary.TAG_UNK:
      tree = trees.InternalParseNode(tags[0], [tree])
  else:
    i = np.argmax(dist)
    tree_l = distance_to_tree(dist[:i], cons[:i], tags[:i + 1], leaves[:i + 1])
    tree_r = distance_to_tree(dist[i + 1:], cons[i + 1:], tags[i + 1:], leaves[i + 1:])
    tree = trees.InternalParseNode(cons[i], [tree_l, tree_r])
  return tree


for tree in test_parse:
  bin_tree = binarize_tree(tree)
  assert str(tree) == str(debinarize_tree(bin_tree)[0])
  leaves = list(bin_tree.leaves())
  dist, cons, tags, _ = tree_to_distance(bin_tree)
  rec_bin_tree = distance_to_tree(dist, cons, tags, leaves)
  assert str(bin_tree) == str(rec_bin_tree)
  assert str(tree) == str(debinarize_tree(rec_bin_tree)[0])


#dist, _, labels = tree_to_distance(tree)
#import ipdb; ipdb.set_trace()
#tree = debinarize_tree(tree)[0]
#print(tree.convert().linearize())
