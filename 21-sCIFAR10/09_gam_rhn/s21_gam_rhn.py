import sys
sys.path.append('../../')


from tframe.utils.script_helper import Helper
from tframe.trainers.trainer import TrainerHub as Config


Helper.register_flags(Config)
s = Helper()
# s.set_python_cmd_suffix()
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
s.register('epoch', 10000)
# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
gpu_id = 0

summ_name = s.default_summ_name

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('rhn_dropout', 0.1, 0.2, 0.3)
s.register('gam_dropout', 0.1, 0.2)
s.register('output_dropout', 0.1, 0.2, 0.3)
s.register('lr', 0.0008, 0.0004, 0.0001, 0.00002)
s.run(1)

