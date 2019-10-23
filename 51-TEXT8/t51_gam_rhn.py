import tensorflow as tf
import txt8_core as core
import txt8_mu as m
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
  console.start('{} on TEXT8 task'.format(model_name.upper()))

  th = core.th
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
  th.hidden_dim = 50

  th.gam_config = '3x750'
  th.head_size = 300

  th.hyper_kernel = 'gru'
  th.state_size = 750
  th.num_layers = 10

  th.gam_dropout = 0.4
  th.rhn_dropout = 0.3
  th.input_dropout = 0.2
  th.output_dropout = 0.2
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.batch_size = 100
  th.num_steps = 100

  th.patience = 5

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0001

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
  # 5. dynamic evaluation
  # ---------------------------------------------------------------------------
  if th.dynamic_evaluation:
    th.train = False
    th.de_max_batches = 450
    th.de_val_pct = 0.1
    th.de_eval_val_set = True
    th.train_stats_exists = False
    th.de_eta = '0.00001'
    th.de_lambda = '0.0075'

  # ---------------------------------------------------------------------------
  # 6. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = GamRHN.mark()
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()
