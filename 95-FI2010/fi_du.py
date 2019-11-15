import numpy as np

from tframe.data.sequences.finance.fi2010 import FI2010
from tframe.data.sequences.seq_set import SequenceSet


def load_data(path):
  # Load training set and test set
  train_set, test_set = FI2010.load(path, horizon=50)
  # For this task, train_set is used for validation
  val_set = get_balanced_seq_set(
    train_set, name='Train*Set', M=min(test_set.structure))
  return train_set, val_set, test_set


def get_balanced_seq_set(seq_set, sections=None, name='Balanced Set', M=None):
  assert isinstance(seq_set, SequenceSet)
  if sections is None:
    if M is None: M = min(seq_set.structure)
    sections = [int(np.ceil(s / M)) for s in seq_set.structure]
  assert isinstance(sections, list)
  features, targets = [], []
  for x, y, s in zip(seq_set.features, seq_set.targets, sections):
    if s == 1:
      features.append(x)
      targets.append(y)
      continue
    L = int(len(x) / s)
    indices = [(i + 1) * L for i in range(s - 1)]
    features += np.split(x, indices)
    targets += np.split(y, indices)
  balanced_set = SequenceSet(features, targets, name=name)
  assert len(balanced_set.structure) == sum(sections)
  assert sum(balanced_set.structure) == sum(seq_set.structure)
  return balanced_set


