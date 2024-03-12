# Learning and Sustaining Shared Normative Systems via Bayesian Rule Induction in Markov Games

With this piece of code, we formalize the problem of constrained norm learning from observing other agents' actions in the context of Markov games in a multi-agent environment via approximately Bayesian rule induction of obligative and prohibitive norms. The environment is based on the [Melting Pot](https://github.com/deepmind/meltingpot) project by DeepMind taking together the different games [Commons Harvest](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/commons_harvest__closed.py), [Clean Up](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/clean_up.py), and [Territory](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/territory.py) to allow agents to learn and sustain a range of norms to handle various dilemmas such as a tragedy of the commons, social-role conditioned labor, and territorial norms.

You can use this code to test your own version of this. The main test scenarios are (1) Pure norm learning from a predefined set of norms, (2) Intergenerational norm transmission, and (3) Spontaneous norm emergence.

<p align="center">
<img src="assets/norm-compliant-planning.jpg" width="320" />
<img src="assets/norm-learning.jpg" width="320" />
<img src="assets/example.gif" width="180" />
</p>

For more information, see [our paper](https://arxiv.org/abs/2402.13399)

> Ninell Oldenburg & Tan Zhi-Xuan. 2024. **Learning and Sustaining Shared Normative Systems via Bayesian Rule Induction in Markov Games**. In Proc. of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2024).

---
## Setup

For this code, your machine requires `Python 3.10` or higher. Follow these steps:

> 1. Go to the [Melting Pot](https://github.com/google-deepmind/meltingpot) repository and follow their installation steps. From their docs: If you get a `ModuleNotFoundError: No module named 'meltingpot.python'` error, you can solve it by exporting the `meltingpot` home directory as PYTHONPATH (e.g. by calling export PYTHONPATH=$(pwd))
> 2. Download the [meltingpot](https://github.com/ninell-oldenburg/social-contracts/tree/main/meltingpot) directory from this repo and replace the `meltingpot` directory of your newly installed Melting Pot instance. Note: make sure the to-be-replaced directory is your local version of [this one](https://github.com/google-deepmind/meltingpot/tree/main/meltingpot) -- their repository structure is something like `meltingpot/meltingpot` so watch out to not confuse these two.
> 3. Done! Now you can run the files.

---
## Repository Structure
