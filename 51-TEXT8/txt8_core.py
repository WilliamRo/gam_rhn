import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
import tensorflow as tf
from tframe import console, SaveMode
from tframe.trainers.trainer import TrainerHub as Config
from tframe import Classifier

import txt8_du as du


from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Config(as_global=True)
th.data_dir = from_root('51-TEXT8/data/')
th.job_dir = from_root('51-TEXT8')
# -----------------------------------------------------------------------------
# Device configurations
# -----------------------------------------------------------------------------
th.allow_growth = True
th.gpu_memory_fraction = 0.90

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [1]
th.target_dim = 1
th.num_classes = 27
th.output_dim = th.num_classes
th.target_dtype = tf.int32

th.val_size = 5000000
th.test_size = 4999999

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.epoch = 1000

th.batch_size = 128
th.validation_per_round = 2
th.export_tensors_upon_validation = True

th.early_stop = True
th.patience = 5

th.num_steps = 100
th.print_cycle = 1
th.gather_note = True

th.save_model = True
th.save_mode = SaveMode.ON_RECORD
th.overwrite = True

th.clip_threshold = 1.0
th.clip_method = 'value'

th.val_batch_size = 20
th.val_num_steps = 5000
th.val_progress_bar = True
th.evaluate_test_set = True
th.eval_batch_size = 1


def activate():
  # Load data
  train_set, val_set, test_set = du.load_data(
    th.data_dir, th.val_size, th.test_size)
  th.train_set = train_set

  # Build model
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Train or evaluate
  if th.train:
    model.train(
      train_set, validation_set=val_set, test_set=test_set, trainer_hub=th)
  elif th.dynamic_evaluation:
    model.evaluate_model(test_set, dynamic=True, val_set=val_set)
  else:
    model.evaluate_model(train_set, batch_size=1)
    model.evaluate_model(val_set, batch_size=1)
    model.evaluate_model(test_set, batch_size=1)

  # End
  model.shutdown()
  console.end()
