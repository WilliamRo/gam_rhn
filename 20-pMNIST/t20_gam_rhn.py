import tensorflow as tf
import pm_core as core
import pm_mu as m
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
  console.start('{} on pMNIST task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.permute = True

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.visible_gpu_id = 0
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.gam_config = '4x60'
  th.head_size = 60

  th.hyper_kernel = 'gru'
  th.state_size = 120
  th.num_layers = 1

  th.gam_dropout = 0.2
  th.rhn_dropout = 0.2
  th.output_dropout = 0.1
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.batch_size = 128
  th.validation_per_round = 10

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0008

  th.clip_threshold = 1.0
  th.clip_method = 'value'
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.train = True
  th.save_model = True
  th.overwrite = False

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '_{}'.format('P' if th.permute else 'NP')
  th.mark = GamRHN.mark() + tail
  th.mark += '_g{}r{}o{}'.format(
    th.gam_dropout, th.rhn_dropout, th.output_dropout)
  th.gather_summ_name = th.prefix + summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()

