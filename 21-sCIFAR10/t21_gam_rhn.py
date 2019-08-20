import scf10_core as core
import scf10_mu as m
import tensorflow as tf
from tframe.utils.misc import date_string
from tframe import console
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
  console.start('{} on sCIFAR-10 task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.centralize_data = False

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.visible_gpu_id = 0
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.allow_growth = True
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.gam_config = '5x60'
  th.head_size = 100

  th.hyper_kernel = 'gru'
  th.state_size = 300
  th.num_layers = 1

  th.head_bias = True
  th.address_bias = True

  th.gam_dropout = 0.1
  th.rhn_dropout = 0.3
  th.output_dropout = 0.25
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.batch_size = 128
  th.validation_per_round = 10

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.0004

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
  th.mark = GamRHN.mark()
  th.mark += '_g{}r{}o{}'.format(
    th.gam_dropout, th.rhn_dropout, th.output_dropout)
  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()
