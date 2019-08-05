from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.sequences.cifar10 import sCIFAR10


def load_data(data_dir):
  train_set, val_set, test_set = sCIFAR10.load(data_dir)
  assert isinstance(train_set, SequenceSet)
  assert isinstance(val_set, SequenceSet)
  assert isinstance(test_set, SequenceSet)
  return train_set, val_set, test_set


