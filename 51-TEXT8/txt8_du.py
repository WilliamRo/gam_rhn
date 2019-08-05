from tframe import checker
from tframe.data.sequences.nlp.text8 import Text8
from tframe.data.dataset import DataSet


def load_data(path, val_size=5000000, test_size=4999999):
  data_sets = Text8.load(path, val_size=val_size, test_size=test_size)
  checker.check_type(data_sets, DataSet)
  return data_sets

