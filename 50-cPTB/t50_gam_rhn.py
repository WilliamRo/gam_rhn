import tensorflow as tf
import cptb_core as core
import cptb_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.hyper.gam_rhn import GamRHN

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gam_rhn'
id = 9
def model(th):
  assert isinstance(th, core.Config)
  cell = GamRHN(
    gam_config=th.gam_config,
    head_size=th.head_size,
    state_size=th.state_size,
    num_layers=th.num_layers,
    kernel=th.hyper_kernel,
    gam_dropout=th.gam_dropout,
    rhn_dropout=th.rhn_dropout,
  )
  return m.typical(th, cell)


def main(_):
  console.start('{} on cPTB task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. data set setup
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.visible_gpu_id = 0
  th.allow_growth = True
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.hidden_dim = 150

  th.gam_config = '3x600'
  th.head_size = 300

  th.hyper_kernel = 'gru'
  th.state_size = 600
  th.num_layers = 5

  th.gam_dropout = 0.4
  th.rhn_dropout = 0.3
  th.input_dropout = 0.3
  th.output_dropout = 0.2
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 128
  th.num_steps = 100

  th.patience = 15

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0004

  th.clip_threshold = 1.0
  th.clip_method = 'value'

  th.validation_per_round = 2
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.overwrite = False

  th.val_num_steps = 1000
  th.val_batch_size = 20

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = GamRHN.mark()
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()


