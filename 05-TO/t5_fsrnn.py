import tensorflow as tf
import to_core as core
import to_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.hyper.fsrnn import FastSlow


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'fsrnn'
id = 5
def model(th):
  assert isinstance(th, m.Config)
  cell = FastSlow(
    fast_size=th.fast_size,
    fast_layers=th.fast_layers,
    slow_size=th.slow_size,
    hyper_kernel=th.hyper_kernel,
    forget_bias=th.forget_bias_initializer,
  )
  return m.typical(th, cell)


def main(_):
  console.start('{} on TO task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.sequence_length = 100
  th.bits = 3

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

  th.fast_layers = 2
  table = {2: 41, 3: 45, 4: 47, 5: 49, 6: 50}
  th.fast_size = table[th.fast_layers]
  th.slow_size = th.fast_size
  th.hyper_kernel = 'lstm'

  th.forget_bias_initializer = 2.0
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.max_iterations = 50000
  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.export_tensors_upon_validation = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '_{}bits_L{}'.format(th.bits, th.sequence_length)
  th.mark = FastSlow.mark() + tail
  th.gather_summ_name = th.prefix + summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()
