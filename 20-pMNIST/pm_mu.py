from tframe import Classifier

from tframe.layers import Input, Activation
from tframe.models import Recurrent
from tframe.layers.common import Dropout
from tframe.layers.advanced import Dense

from tframe.configs.config_base import Config


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
  assert isinstance(model, Classifier)
  assert isinstance(th, Config)
  # Add dropout if necessary
  if th.output_dropout > 0: model.add(Dropout(1 - th.output_dropout))
  # Add output layer
  model.add(Dense(num_neurons=th.output_dim))
  model.add(Activation('softmax'))

  model.build(last_only=True, metric=['accuracy', 'loss'],
              batch_metric='accuracy', eval_metric='accuracy')
