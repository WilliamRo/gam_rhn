from tframe.data.sequences.nlp.ptb import PTB


def load_data(path):
  return PTB.load(path, level='char')


if __name__ == '__main__':
  from cptb_core import th
  train_set, valid_set, test_set = load_data(th.data_dir)
  print()
