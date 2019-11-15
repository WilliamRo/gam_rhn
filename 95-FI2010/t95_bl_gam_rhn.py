import tensorflow as tf
import fi_core as core
import fi_mu as m
from tframe import console
from tframe.utils.misc import date_string

from tframe.nets.hyper.gam_rhn import GamRHN
from tframe.layers.hyper.bilinear import Bilinear
from tframe.layers.common import Flatten
from tframe.nets.rnn_cells.conveyor import Conveyor
from tframe.layers.common import Dropout

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'bl_gam_rhn'
id = 1
def model(th):
  assert isinstance(th, m.Config)
  layers = [Conveyor(length=th.conveyor_length)]
  # Bilinear part
  bl_config = m.bl_config()
  for d1, d2 in bl_config[:-1]:
    layers.append(Bilinear(d1, d2, 'relu', max_norm=th.max_norm))
    if th.dropout > 0: layers.append(Dropout(1 - th.dropout))

  d1, d2 = bl_config[-1]
  layers.append(Bilinear(d1, d2, 'tanh', max_norm=th.max_norm))
  layers.append(Flatten())

  # GAM-RHN part
  layers.append(GamRHN(
    gam_config=th.gam_config,
    head_size=th.head_size,
    state_size=th.state_size,
    num_layers=th.num_layers,
    kernel=th.hyper_kernel,
    gam_dropout=th.gam_dropout,
    rhn_dropout=th.rhn_dropout,
  ))
  return m.typical(th, layers)


def main(_):
  console.start('{} on FI-2010 task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  th.prefix = '{}_'.format(date_string())
  summ_name = model_name
  th.visible_gpu_id = 0

  th.suffix = ''
  # ---------------------------------------------------------------------------
  # 1. dataset setup
  # ---------------------------------------------------------------------------
  th.volume_only = True

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.conveyor_length = 15
  th.conveyor_input_shape = [20]

  # - - - - - - - - - - - - - - - - - - - - - - ↑ common - - ↓ model specific
  # Bilinear part
  th.archi_string = '20x60+15x80+10x100+5x120+5x4'
  th.max_norm = 2.5

  # GAM-RHN part
  th.gam_config = '2x50'
  th.head_size = 15

  th.hyper_kernel = 'gru'
  th.state_size = 200
  th.num_layers = 8

  th.dropout = 0.2
  th.gam_dropout = 0.3
  th.rhn_dropout = 0.5
  th.output_dropout = 0.5
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.batch_size = 64
  th.sub_seq_len = 5000
  th.num_steps = 10

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  th.clip_threshold = 1.0
  th.clip_method = 'value'

  th.validation_per_round = 2

  th.lives = 5
  th.lr_decay = 0.4
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}_{}'.format(th.archi_string, GamRHN.mark())
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()