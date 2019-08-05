import sys, os
ROOT = os.path.abspath(__file__)
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe import console
from tframe.models.sl.classifier import Classifier
from tframe.enums import SaveMode
from tframe.trainers.trainer import TrainerHub as Config

import pm_du as du


from_root = lambda path: os.path.join(ROOT, path)

# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Config(as_global=True)
th.data_dir = from_root('20-pMNIST/data')
th.job_dir = from_root('20-pMNIST')
# -----------------------------------------------------------------------------
# Some device configurations
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.45

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.input_shape = [1]
th.output_dim = 10
th.permute = False

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.epoch = 10000
th.batch_size = 128
th.validation_per_round = 10
th.export_tensors_upon_validation = True

th.early_stop = True
th.patience = 15

th.num_steps = -1
th.shuffle = True
th.print_cycle = 1
th.gather_note = True

th.save_model = True
th.save_mode = SaveMode.ON_RECORD
th.overwrite = False

th.clip_threshold = 1.0
th.clip_method = 'value'
th.val_batch_size = 5000
th.eval_batch_size = 5000

# -----------------------------------------------------------------------------
# Set some default model configs
# -----------------------------------------------------------------------------

def activate():
  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Load data
  train_set, val_set, test_set = du.load_data(
    th.data_dir, permute=th.permute, permute_mark='alpha')

  # Train or evaluate
  if th.train: model.train(
    train_set, validation_set=val_set, trainer_hub=th, test_set=test_set)
  else:
    # Evaluate model
    model.evaluate_model(train_set, batch_size=5000)
    model.evaluate_model(val_set, batch_size=5000)
    model.evaluate_model(test_set, batch_size=5000)

  # End
  model.shutdown()
  console.end()
