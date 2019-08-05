from tframe.data.sequences.seq_set import SequenceSet
from tframe.data.sequences.pmnist import pMNIST


def load_data(data_dir, permute, permute_mark='alpha'):
  train_set, val_set, test_set = pMNIST.load(
    data_dir, permute=permute, permute_mark=permute_mark)
  assert isinstance(train_set, SequenceSet)
  assert isinstance(val_set, SequenceSet)
  assert isinstance(test_set, SequenceSet)
  return train_set, val_set, test_set


