# Noise Conditioned Score Networks for Motion Trajectories

This repo contains an implementation of [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) modified for use with imitation learning algorithms like noise conditioned energy based annealed rewards. The underlying algorithm is the same as in the original repo. It is however modified for use with demonstration motion data.



## Dependencies

* PyTorch
* PyYAML
* tqdm
* pillow
* tensorboardX
* seaborn

## Running Experiments

### Project Structure

`main.py` is the common gateway to all experiments. Type `python main.py --help` to get its usage description.

```bash
usage: main.py [-h] [--runner RUNNER] [--config CONFIG] [--seed SEED]
               [--run RUN] [--doc DOC] [--comment COMMENT] [--verbose VERBOSE]
               [--test] [--resume_training] [-o IMAGE_FOLDER]

optional arguments:
  -h, --help            show this help message and exit
  --runner RUNNER       The runner to execute
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --run RUN             Path for saving running related data.
  --doc DOC             A string for documentation purpose
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --test                Whether to test the model
  --resume_training     Whether to resume training
  -o IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The directory of image outputs
```

The main runner class, used for annealed langevin dynamics is AnnealRunner.

* `AnnealRunner` The main runner class for experiments related to NCSN and annealed Langevin dynamics.

Configuration files are stored in  `configs/`. For example, the configuration file of `AnnealRunner` is `configs/anneal.yml`. Log files are commonly stored in `run/logs/doc_name`, and tensorboard files are in `run/tensorboard/doc_name`. Here `doc_name` is the value fed to option `--doc`. However, if `main.py` is run coupled with other algorithms, a config can also be passed via the primary algorithm.


### Training

The usage of `main.py` is quite self-evident. For example, we can train an NCSN by running

```bash
python main.py --runner AnnealRunner --config anneal.yml --doc cifar10
```

Then the model will be trained according to the configuration files in `configs/anneal.yml`. The log files will be stored in `run/logs/cifar10`, and the tensorboard logs are in `run/tensorboard/cifar10`.

### Sampling

Suppose the log files are stored in `run/logs/cifar10`. We can produce samples to folder `samples` by running

```bash
python main.py --runner AnnealRunner --test -o samples
```


