import math
import numpy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from blocks import WeightDrop, embedded_dropout


class Shuffle(nn.Module):
  def __init__(self, permutation, contiguous=True):
    super(Shuffle, self).__init__()
    self.permutation = permutation
    self.contiguous = contiguous

  def forward(self, input):
    shuffled = input.permute(*self.permutation)
    if self.contiguous:
      return shuffled.contiguous()
    else:
      return shuffled


class LayerNormalization(nn.Module):
  ''' Layer normalization module '''

  def __init__(self, d_hid, eps=1e-3):
    super(LayerNormalization, self).__init__()

    self.eps = eps
    self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
    self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

  def forward(self, z):
    if z.size(1) == 1:
      return z

    mu = torch.mean(z, keepdim=True, dim=-1)
    sigma = torch.std(z, keepdim=True, dim=-1)
    ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
    ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
    return ln_out


class DistanceParser(nn.Module):
  def __init__(self,
         vocab_size, embed_size, hid_size, tag_size, label_size, wordembed=None,
         dropout=0.2, dropoute=0.1, dropoutr=0.1, nlayers=2, use_transformer=False):
    super(DistanceParser, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.tag_size = tag_size
    self.hid_size = hid_size
    self.label_size = label_size
    self.drop = nn.Dropout(dropout)
    self.dropout = dropout
    self.dropoute = dropoute
    self.dropoutr = dropoutr
    self.use_transformer = use_transformer
    self.encoder = nn.Embedding(vocab_size, embed_size)
    self.tag_encoder = nn.Embedding(tag_size, embed_size)
    if wordembed is not None:
      self.encoder.weight.data = torch.FloatTensor(wordembed)

    if not use_transformer:
      self.word_rnn = nn.LSTM(
          2 * embed_size, hid_size, num_layers=nlayers, batch_first=True, dropout=dropout,
          bidirectional=True)
      self.word_rnn = WeightDrop(
          self.word_rnn,
          ['weight_hh_l%d' % l for l in range(nlayers)],
          dropout=dropoutr)
      self.conv1 = nn.Sequential(
          nn.Dropout(dropout),
          nn.Conv1d(hid_size * 2,
                    hid_size, 2),
          nn.BatchNorm1d(hid_size),
          nn.ReLU())
      # label rnn
      self.label_rnn = nn.LSTM(
          hid_size, hid_size,
          num_layers=nlayers, batch_first=True, dropout=dropout,
          bidirectional=True)
      self.label_rnn = WeightDrop(
          self.label_rnn, ['weight_hh_l%d' % l for l in range(nlayers)],
          dropout=dropoutr)
    else:
      assert False
    # predicting unary chains ending in a terminal
    self.unary_out = nn.Sequential(
      nn.Linear(hid_size * 2, label_size)
    )
    # predict syntactic distance
    self.dist_out = nn.Sequential(
      nn.Linear(hid_size * 2, 1),
    )
    # predict constituency label
    self.label_out = nn.Sequential(
      nn.Linear(hid_size * 2, label_size)
    )

  def feature_drop(self, x):
    if self.training:
      B, _, D = x.size()
      dmask = torch.ones(B, D)
      dmask.bernoulli_(1. - self.dropout).div(1. - self.dropout).float()
      if x.is_cuda:
        dmask = dmask.cuda()
      x = x * dmask[:, None, :]
    return x

  def forward(self, words, tags):
    mask = (words > 0).float()
    B, T = words.size()
    emb_words = embedded_dropout(self.encoder, words, dropout=self.dropoute if self.training else 0)
    emb_words = self.feature_drop(emb_words)
    emb_tags = embedded_dropout(self.tag_encoder, tags, dropout=self.dropoute if self.training else 0)
    emb_tags = self.feature_drop(emb_tags)
    emb_plus_tag = torch.cat([emb_words, emb_tags], dim=-1)
    if not self.use_transformer:
      sent_lengths = mask.sum(1).cpu().numpy().astype('int')
      rnn_word_out, _ = self.word_rnn(emb_plus_tag)
      conv_out = self.conv1(rnn_word_out.permute(0, 2, 1)).permute(0, 2, 1)
      rnn_top_out, _ = self.label_rnn(conv_out)
      dist_out = self.feature_drop(rnn_top_out)
      label_out = dist_out
      unary_out = self.feature_drop(rnn_word_out)
    else:
      dist_out = self.transf(emb_plus_tag)
      label_out = dist_out
      unary_out = dist_out

    unary_pred = self.unary_out(unary_out.contiguous().view(-1, self.hid_size * 2))
    dist_pred = self.dist_out(dist_out).squeeze(dim=-1)  # (bsize, ndst)
    label_pred = self.label_out(label_out)                # (bsize, ndst, arcsize)
    return (dist_pred,
            label_pred.contiguous().view(-1, self.label_size),
            unary_pred.contiguous().view(-1, self.label_size))
