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
         dropout=0.2, dropoute=0.1, dropoutr=0.1):
    super(DistanceParser, self).__init__()
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.tag_size = tag_size
    self.hid_size = hid_size
    self.label_size = label_size
    self.drop = nn.Dropout(dropout)
    self.dropoute = dropoute
    self.dropoutr = dropoutr
    self.encoder = nn.Embedding(vocab_size, embed_size)
    if wordembed is not None:
      self.encoder.weight.data = torch.FloatTensor(wordembed)

    self.tag_encoder = nn.Embedding(tag_size, embed_size)
    self.word_rnn = nn.LSTM(
        2 * embed_size, hid_size, num_layers=2, batch_first=True, dropout=dropout,
        bidirectional=True)
    self.word_rnn = WeightDrop(
        self.word_rnn,
        ['weight_hh_l0', 'weight_hh_l1'],
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
        num_layers=2, batch_first=True, dropout=dropout,
        bidirectional=True)
    self.label_rnn = WeightDrop(
        self.label_rnn, ['weight_hh_l0'], dropout=dropoutr)

    # predicting unary chains ending in a terminal
    self.unary_out = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(hid_size * 2, hid_size),
      nn.BatchNorm1d(hid_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hid_size, label_size)
    )

    # predict syntactic distance
    self.dist_out = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(hid_size * 2, hid_size),
      nn.BatchNorm1d(hid_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hid_size, 1),
      nn.BatchNorm1d(1),
    )

    # predict constituency label
    self.label_out = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(hid_size * 2, hid_size),
      nn.BatchNorm1d(hid_size),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hid_size, label_size),
    )

  def forward(self, words, tags):
    mask = (words > 0).float()
    bsz, ntoken = words.size()
    emb_words = embedded_dropout(self.encoder, words, dropout=self.dropoute if self.training else 0)
    emb_words = self.drop(emb_words)

    emb_tags = embedded_dropout(self.tag_encoder, tags, dropout=self.dropoute if self.training else 0)
    emb_tags = self.drop(emb_tags)

    def run_rnn(input, rnn, lengths):
      sorted_idx = numpy.argsort(lengths)[::-1].tolist()
      rnn_input = pack_padded_sequence(input[sorted_idx], lengths[sorted_idx], batch_first=True)
      rnn_out, _ = rnn(rnn_input)  # (bsize, ntoken, hidsize*2)
      rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
      rnn_out = rnn_out[numpy.argsort(sorted_idx).tolist()]
      return rnn_out

    sent_lengths = mask.sum(1).cpu().numpy().astype('int')
    emb_plus_tag = torch.cat([emb_words, emb_tags], dim=-1)
    rnn_word_out = run_rnn(emb_plus_tag, self.word_rnn, sent_lengths)
    unary_pred = self.unary_out(rnn_word_out.view(-1, self.hid_size * 2))

    conv_out = self.conv1(rnn_word_out.permute(0, 2, 1)).permute(0, 2, 1)  # (bsize, ndst, hidsize)
    rnn_top_out = run_rnn(conv_out, self.label_rnn, sent_lengths - 1)
    rnn_top_out = rnn_top_out.view(-1, self.hid_size * 2)

    dist_pred = self.dist_out(rnn_top_out).squeeze(dim=-1)  # (bsize, ndst)
    label_pred = self.label_out(rnn_top_out)                # (bsize, ndst, arcsize)
    return (dist_pred,
            label_pred.contiguous().view(-1, self.label_size),
            unary_pred.contiguous().view(-1, self.label_size))