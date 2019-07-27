from tframe import Classifier

from tframe.layers import Input, Linear, Activation
from tframe.models import Recurrent

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
  # Add output layer
  model.add(Linear(
    output_dim=th.output_dim,
    use_bias=th.bias_out_units,
  ))
  model.add(Activation('softmax'))

  model.build(metric='accuracy', batch_metric='accuracy', last_only=True)
