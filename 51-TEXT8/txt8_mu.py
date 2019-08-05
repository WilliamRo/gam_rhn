import tensorflow as tf
from tframe import Classifier

from tframe.layers import Input, Activation
from tframe.layers.common import Dropout
from tframe.models import Recurrent

from tframe.trainers.trainer import TrainerHub as Config

from tframe.layers.common import Onehot
from tframe.layers.advanced import Dense



def typical(th, cells):
  assert isinstance(th, Config)

  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)

  # Add layers
  model.add(Input(sample_shape=th.input_shape, dtype=tf.int32))
  model.add(Onehot(depth=th.num_classes))
  emb_init = tf.initializers.random_uniform(-1, 1)
  model.add(Dense(th.hidden_dim, use_bias=False, weight_initializer=emb_init))

  if th.input_dropout > 0: model.add(Dropout(1 - th.input_dropout))
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

  model.build(loss='cross_entropy', metric='bpc', batch_metric='bpc')


