# Deep Feature Extraction for Sample-Efficient Reinforcement Learning

This is an implementation in Keras and Scikit-learn of the approach
described in [my Master's thesis](https://github.com/danielegrattarola/thesis).

This work was supervised by Prof. Marcello Restelli, Dott. Carlo D'Eramo,
and Matteo Pirotta, Ph.D. from Politecnico di Milano.

Please cite the thesis if you use any of this code for your work:
```
@article{grattarola2017deep,
  title={Deep Feature Extraction for Sample-Efficient Reinforcement Learning},
  author={Grattarola, Daniele},
  year={2017},
  publisher={Politecnico di Milano},
  url={https://www.politesi.polimi.it/handle/10589/136065}
}
```

## Abstract
Deep reinforcement learning (DRL) has been under the spotlight of
artificial intelligence research in recent years, enabling reinforcement
learning agents to solve control problems that were previously
considered intractable. The most effective DRL methods, however,
require a great amount of training samples (in the order of tens of
millions) in order to learn good policies even on simple environments,
making them a poor choice in real-world situations where the collection
of samples is expensive.
In this work, we propose a sample-efficient DRL algorithm that combines
unsupervised deep learning to extract a representation of the
environment, and batch reinforcement learning to learn a control policy
using this new state space. We also add an intermediate step of feature
selection on the extracted representation in order to reduce the
computational requirements of our agent to the minimum. We test our
algorithm on the Atari games environments, and compare the performance
of our agent to that of the DQN algorithm by Mnih et al. (2015). We
show that even if the final performance of our agent amounts to a
quarter of DQN’s, we are able to achieve good sample efficiency and a
better performance on small datasets.

## Installation
To run the code in this repo, the IFQI module by @teopir is required.
This is available [on GitHub](https://github.com/teopir/ifqi), but the
development of the package was already stalled when I was working on the
thesis, so you might have troubles with it.
Let me know if I can help.

To install, clone the repository and install with pip:
```sh
git clone https://github.com/danielegrattarola/nips2017-deepIFS.git
cd nips2017-deepIFS
sudo pip install -e .
```

## Running
Basically the only script you need to run is `scripts/run_main.py`, which
can execute a complete run of the algorithm including the training of
the AE, the feature selection, and FQI.

To run the algorithm exactly as desccribed in the thesis, run the
following command:
```sh
$ python run_main.py \
    --main-alg-iters 10 \
    --train-ae \
    --binarize \
    --fs \
    --rfs \
    --fqi-no-ar
```
Note that you may have to play with the datasets sizes in order to make
the whole thing run on your machine.

**Even bigger note**: this code is released under the [CRAPL](http://matt.might.net/articles/crapl/CRAPL-LICENSE.txt)
license and ~~will~~ may not work as intended in ~~most~~ some cases.