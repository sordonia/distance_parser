
import numpy as np
import trees
import vocabulary


def binarize_tree(tree):
  """Binarizes a tree by choosing the leftmost split point.
  """
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
  """Debinarizes the tree.
  """
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
      # handle the unary chains here
      assert len(root.children) == 1
      t = [root.label]
    else:
      # just predict a UNK label
      assert isinstance(root, trees.LeafParseNode)
      t = [vocabulary.UNK]
  return d, c, t, h


def distance_to_tree(dist, cons, unary, leaves):
  if dist == []:
    tree = leaves[0]
    if unary[0] != vocabulary.TAG_UNK:
      tree = trees.InternalParseNode(unary[0], [tree])
  else:
    i = np.argmax(dist)
    tree_l = distance_to_tree(dist[:i], cons[:i], unary[:i + 1], leaves[:i + 1])
    tree_r = distance_to_tree(dist[i + 1:], cons[i + 1:], unary[i + 1:], leaves[i + 1:])
    tree = trees.InternalParseNode(cons[i], [tree_l, tree_r])
  return tree

