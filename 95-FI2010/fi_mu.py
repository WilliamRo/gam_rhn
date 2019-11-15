import tensorflow as tf
from tframe import Classifier

from tframe.layers import Input, Activation
from tframe.layers.common import Dropout
from tframe.models import Recurrent

from tframe.layers.advanced import Dense
from tframe.trainers import SmartTrainerHub as Config

from tframe.data.sequences.finance.fi2010 import Extract
from tframe.layers.specific.lob_extraction import Significance

from tframe.nets.rnn_cells.conveyor import Conveyor
from tframe.layers.common import Flatten, Reshape


def typical(th, cells):
  assert isinstance(th, Config)

  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)
  # Add layers
  model.add(Input(sample_shape=th.input_shape))

  # Add hidden layers
  if not isinstance(cells, (list, tuple)): cells = [cells]
  for cell in cells: model.add(cell)
  # Build model and return
  output_and_build(model, th)
  return model


def output_and_build(model, th):
  assert isinstance(model, Classifier) and isinstance(th, Config)
  # Add output dropout if necessary
  if th.output_dropout > 0: model.add(Dropout(1 - th.output_dropout))
  # Add output layer
  model.add(Dense(num_neurons=th.output_dim))
  model.add(Activation('softmax'))

  # Build model
  model.build(loss=th.loss_string, metric=['loss', 'f1'], batch_metric='f1')


def bl_config():
  """e.g. th.archi_string = '10x60+5x120+1x20'"""
  from fi_core import th
  layers = th.archi_string.split('+')
  configs = []
  for layer in layers:
    sizes = [int(s) for s in layer.split('x')]
    assert len(sizes) == 2 and sizes[0] > 0 and sizes[1] > 0
    configs.append(sizes)
  assert len(configs) > 0
  return configs


