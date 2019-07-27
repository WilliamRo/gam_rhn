from tframe.data.perpetual_machine import PerpetualMachine
from tframe.data.sequences.to import TO
from tframe.data.sequences.seq_set import SequenceSet


def load(data_dir, L, N, fixed_length=True, val_size=200, test_size=1000):
  train_set, val_set, test_set = TO.load(
    data_dir, val_size, test_size, L, N, fixed_length)
  assert isinstance(train_set, PerpetualMachine)
  assert isinstance(val_set, SequenceSet)
  assert isinstance(test_set, SequenceSet)
  return train_set, val_set, test_set



