import tensorflow as tf
import to_core as core
import to_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.hyper.rhn import RHN


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'rhn'
id = 4
def model(th):
  assert isinstance(th, m.Config)
  cell = RHN(
    state_size=th.state_size,
    num_layers=th.num_layers,
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

  th.num_layers = 1
  table = {1: 96, 2: 90, 3: 88, 4: 87, 5: 87, 6: 86}
  th.state_size = table[th.num_layers]
  th.hyper_kernel = 'ugrnn'

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
  th.mark = RHN.mark() + tail
  th.gather_summ_name = th.prefix + summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()
