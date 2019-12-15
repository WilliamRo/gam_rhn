[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recurrent-highway-networks-with-grouped/language-modelling-on-penn-treebank-character)](https://paperswithcode.com/sota/language-modelling-on-penn-treebank-character?p=recurrent-highway-networks-with-grouped)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recurrent-highway-networks-with-grouped/language-modelling-on-text8)](https://paperswithcode.com/sota/language-modelling-on-text8?p=recurrent-highway-networks-with-grouped)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/recurrent-highway-networks-with-grouped/stock-trend-prediction-on-fi-2010)](https://paperswithcode.com/sota/stock-trend-prediction-on-fi-2010?p=recurrent-highway-networks-with-grouped)

GAM-RHN
=========
Experiments in Recurrent Highway Networks with Grouped Auxiliary Memory paper.
All experiments are done using [tframe](https://github.com/WilliamRo/tframe), which contains a number of neural network APIs based on ```tensorflow```.

![fig1](https://github.com/WilliamRo/gam_rhn/blob/master/figures/gam_rhn_diagram.png?raw=true)

*Figure 1: A diagram for the proposed GAM-RHN architecture.*

**Requirements**

tensorflow (>=1.8.0) and other common packages

**TL;DR**

Change directory into `XX-YYYY` folder (e.g. `50-cPTB`) and run `tX_ZZZ.py` directly. For example,

```shell
william@alienware:~/gam_rhn/50-cPTB$ python t50_gam_rhn.py
```

**More details**

- [File Organization](#file-organization)
- [Trial Configurations](#trial-configurations)
- Other details will be added soon ...

## File Organization

This repository follows a recommended way to organization projects based on `tframe`.

The root folder contains

- Several `XX-YYYY` folders. Here `XX` denotes dataset ID and `YYYY` denotes dataset name. For example `05-TO` contains all necessary files for experiments on Temporal Order problems.
- One `tframe` folder. A submodule used in this repo.
- One `view_notes.py` module. Run this module directly to open a tool to view results of trials with different hyper-parameters. Details will be given later.
- Other unimportant file like `README.md`.

Each `XX-YYYY` folder contains:

- One `data` folder. Data for `YYYY` dataset will be generated or downloaded to this folder.
- Three utility modules
  - `YY_du.py` contains code for loading data as `tframe` dataset.
  - `YY_mu.py` contains utilities for building a `tframe` model.
  - `YY_core.py` contains
    - Code to add paths to `sys.path`
    - Code to instantiate `Config`. 
      - `th = Config(as_global=True)`
    - Some common configurations for all task modules inside `XX-YYYY` including 
      - data input shape `th.input_shape`
      - GPU configurations, e.g. `th.gpu_memory_fraction = 0.45`
      - Hyper-parameters which are likely to be fixed, e.g. `th.batch_size = 64` 
    - `activate` method, contains
      - Code to initiate a `trame` model
      - Code to load data
      - Code for training/evaluation
- Several task modules `tXX_ZZZ.py`. Here `ZZZ` is the model name, e.g. `t21_gdu.py`.  Each task module contains 
  - A method to define the corresponding `tframe` model. 
  - A `main` method containing a list of configurations. At the end of the `main` method, `activate` method defined in `YY_core.py` module will be called.
  - An `if __name__ == '__main__'` block to invoke `tf.app.run()`.
- Several working folder `XX_ZZZ` each of which corresponds to a task module `tXX_ZZZ.py`. Each of these folder mainly contains
  - `tensorflow` checkpoints if `th.save_model == True`
  - `tensorflow` log files if `th.summary == True`
  - `tframe` summary files that can be open via `view_notes.py`
  - A script file `sXX_ZZZ.py` for tuning hyper-parameters. For example, `python s21_gam_rhn.py --lr=0.001,0.0005 --batch_size=128,64,32` 

## Trial Configurations

Trials are configurated using `tframe` `Config` class. The instance name is usually `th` since it's easy to type. 

`tframe` provides hundreds of configuration options for users which are maintained in `tframe/configs` package. Each trial is likely to involve dozens of configurations including dataset configurations, device configurations, console output configurations, path specifications and most importantly, hyper-parameters. To be honestly, this makes `tframe` complicated and forbidding, even the developer himself will be confused sometimes. However, life will be much easier if one is familiar with those options since many of them enable users to carry out very complicated logics by turning on simple options.

**For IDE users**, set the configurations in the corresponding task module `tXX_ZZZ.py` and run this module directly.

Configurations may also be passed via **command line arguments**, e.g.

```powershell
william@alienware:~/gam_rhn/50-cPTB$ python t50_gam_rhn.py --lr=0.0004 --batch_size=128 --state_size=1000
```

In this case configs specified inside `t50_gam_rhn.py` task module

```python
...
def main(_):
    ...
    th.learning_rate = 0.0002
    th.batch_size = 256
    th.state_size = 800
    ...
...
```

will be ignored since command line arguments are of highest priority.

Hyper-parameters can be tuned using a script module. A template can be found in `gam_rhn/50-cPTB/09_gam_rhn/s50_gam_rhn.py`. Script modules must be named to `sXX_ZZZ.py` and put inside the working folder of the corresponding task module `tXX_ZZZ.py`. It will automatically find its corresponding task module and call it several times for every hyper-parameters combination. `tframe` script modules allow users to register hyper-parameters via command line arguments, e.g. 

```shell
william@alienware:~/gam_rhn/50-cPTB/09_gam_rhn$ python s50_gam_rhn.py --lr=0.0004,0.0002 --state_size=1000,2000
```

Note that hyper-parameters specified are separated using commas. Similarly, command line hyper-parameters will overwrite the hyper-parameters specified inside the script module

```python
...
s = Helper()
...
s.register('lr', 0.0008, 0.001)
s.register('state_size', 500, 800)
...
```








