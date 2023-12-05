# Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning

ReImplementation of the GrBAL method of [Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347).
The code is written in Python 3 and builds on Pytorch.

## Getting Started
### A. Anaconda
If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#windows)

``` conda env create -f environment.yml ```

Then, separately install Pytorch 2.1.1 on the newly created environment:
``` conda activate learning_to_adapt ```
``` pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ```

## Usage.
In order to train our reimplementation, run the following command:

```python run_grbal_cp.py ```

When running experiments, the data will be stored in ``` data/$EXPERIMENT_NAME ```. You can visualize the learning process
by using the visualization kit:

``` python viskit/frontend.py data/ ```

In order to visualize and test a learned policy run:

``` python experiment_utils/sim_policy data/```

To train/test the alternate models run:

```python experiments/exps.py```

Note: Before running the code make sure to set the `TRAIN` parameter to `true` if you would like to train the model, `false` otherwise.
To change the model set the `ALGO` variable in the exps.py file.


## Acknowledgements
This repository is based on the original authors' repository: [Author's repository](https://github.com/iclavera/learning_to_adapt).