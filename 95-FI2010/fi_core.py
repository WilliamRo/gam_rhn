import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
import numpy as np
import tensorflow as tf
from tframe import console, SaveMode
from tframe import Classifier
from tframe.trainers import SmartTrainerHub as Config

import fi_du as du

from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Config(as_global=True)
th.data_dir = from_root('95-FI2010/data/')
th.job_dir = from_root('95-FI2010')
# -----------------------------------------------------------------------------
# Device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.3

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.max_level = 10
th.volume_only = True
th.developer_code = 'use_log'

th.input_shape = [20]
th.num_classes = 3
th.output_dim = 3
th.target_dim = 1
th.target_dtype = tf.int32

th.val_size = 1

th.loss_string = 'cross_entropy'

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.epoch = 1000

th.gather_note = True
th.export_tensors_upon_validation = True
th.train = True
th.overwrite = True
th.save_model = True
th.save_mode = SaveMode.ON_RECORD

th.validation_per_round = 2
th.early_stop = True
th.patience = 10
th.lives = 1

th.clip_threshold = 1.0
th.clip_method = 'value'

th.print_cycle = 50
th.val_progress_bar = True

th.val_batch_size = -1
th.val_num_steps = 2000

th.clip_nan_protection = False
th.state_nan_protection = False
th.terminate_on_nan = True

th.lr_decay = 0.4
th.sub_seq_len = 5000

th.use_conveyor = True
th.conveyor_length = 15

def activate():
  # Load datasets
  train_set, val_set, test_set = du.load_data(th.data_dir)
  # Calculate class weights
  if th.class_weights is None and th.loss_string == 'wce':
    train_targets = train_set.stack.targets.flatten()
    samples_per_class = [sum(train_targets == c) for c in range(th.num_classes)]
    class_weights = min(samples_per_class) / np.array(samples_per_class)
    th.class_weights = class_weights
    console.show_status(
      'Class weights set to {}'.format(th.class_weights), '++')

  # Set input shape according to th.max_level and th.volume_only
  du.FI2010.set_input_shape()

  # Build model
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Train or evaluate
  if th.train: model.train(
    train_set, validation_set=val_set, test_set=test_set, trainer_hub=th,
    evaluate=lambda t: du.FI2010.evaluate(t, test_set))
  else: du.FI2010.evaluate(model, test_set)

  # End
  model.shutdown()
  console.end()
