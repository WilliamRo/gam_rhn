import tensorflow as tf
import to_core as core
import to_mu as m
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.lstms import BasicLSTMCell


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'lstm'
id = 1

def model(th):
  assert isinstance(th, m.Config)
  cell = BasicLSTMCell(
    state_size=th.state_size,
    forget_bias_initializer=th.forget_bias_initializer
  )
  return m.typical(th, cell)


def main(_):
  console.start('{} on TO task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.sequence_length = 200
  th.bits = 3

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.job_dir += '/{:02d}_{}'.format(id, model_name)
  summ_name = model_name
  th.visible_gpu_id = 0
  prefix = '{}_'.format(date_string())
  suffix = ''

  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.state_size = 67

  th.forget_bias_initializer = 2.0
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.max_iterations = 10000
  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001

  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.export_tensors_upon_validation = True

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '_{}bits_L{}'.format(th.bits, th.sequence_length) + suffix
  th.mark = prefix + '{}({})'.format(model_name, th.state_size) + tail
  th.gather_summ_name = prefix + summ_name + tail + '.sum'
  core.activate()


if __name__ == '__main__':
  tf.app.run()