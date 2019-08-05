import scf10_core as core
import scf10_mu as m
import tensorflow as tf
from tframe.utils.misc import date_string
from tframe import console

from tframe.nets.hyper.gdu_h import GDU


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gdu'
id = 8
def model(th):
  assert isinstance(th, m.Config)
  cell = GDU(
    configs=th.gdu_string,
    use_reset_gate=th.use_reset_gate,
    dropout=th.dropout,
  )
  return m.typical(th, cell)


def main(_):
  console.start('{} on sCIFAR-10 task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.gdu_string = '5x60'
  th.use_reset_gate = True
  th.sog_version = 1   # sog_v1 is much faster

  th.dropout = 0.1
  th.output_dropout = 0.2
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.batch_size = 128
  th.validation_per_round = 10

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

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
  th.mark = GDU.mark()

  th.mark += '_rdp{}odp{}gc{}'.format(
    th.dropout, th.output_dropout, th.clip_threshold)

  th.gather_summ_name = th.prefix + summ_name + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()

