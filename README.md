# Uncovering Cross-Objective Interference in Multi-Objective Alignment

<p align="center">
  <img src="https://img.shields.io/github/stars/yining610/ctwa?style=social"/>
  <img src="https://img.shields.io/github/forks/yining610/ctwa?style=social"/>
  <img src="https://img.shields.io/github/license/yining610/ctwa?style=flat&v=2"/>
</p>

<p align="center">
  <b>TL;DR: We analyze cross-objective interference in multi-objective alignment of LLMs and introduce an approach to mitigate it.</b><br>
  <a href=""><b>Paper on arXiv</b></a>
</p>

## Folder Structure
```
scripts/     // callable scripts for data preprocessing
verl/        // source code of models, algorithms, data structures, metrics, etc. 
examples/    // bash scripts to run jobs
data/        // pre-processed data used in experiments
```

## Environment 
We use the `Dockerfile` to build the environment. For more setup instructions, please refer to the [verl environment setup guide](https://verl.readthedocs.io/en/latest/start/install.html)

We use Wandb to log experiments, so please log in before running them.

## Experiment
We implemented a total of **8** scalarization algorithms from MOO and MTL, including our CTWA. They are:
1. CTWA (ours): `verl/trainer/ppo/ray_trainer_covariance.py`
2. [Dynamic Reweighting](https://arxiv.org/abs/2509.11452): `verl/trainer/ppo/ray_trainer_dynamic.py`
3. [GradNorm](https://arxiv.org/abs/1711.02257): `verl/trainer/ppo/ray_trainer_gradnorm.py`
4. [Lagrangian Primal-Dual Method](https://papers.nips.cc/paper_files/paper/2013/hash/3493894fa4ea036cfc6433c3e2ee63b0-Abstract.html): `verl/trainer/ppo/ray_trainer_lagrangian.py`
5. [MGDA](https://www.sciencedirect.com/science/article/pii/S1631073X12000738): `verl/trainer/ppo/ray_trainer_mgda.py`
6. Linear: `verl/trainer/ppo/ray_trainer_multiobjective.py`
7. [PAMA](https://arxiv.org/abs/2508.07768): `verl/trainer/ppo/ray_trainer_pama.py`
8. [Tchebycheff Scalarization](https://link.springer.com/chapter/10.1007/978-3-642-87563-2_5): `verl/trainer/ppo/ray_trainer_tchebycheff.py`

We provide the bash script of each algorithm in the `examples/` directory. Taking an example of training Qwen2.5-1.5B-Base using CTWA:
```
bash examples/ctwa_trainer/run_covariance_math.sh
```