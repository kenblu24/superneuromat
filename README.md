# SuperNeuroMAT

SuperNeuroMAT is a Python package for simulating and analyzing spiking neural networks.

Documentation available: https://kenblu24.github.io/superneuromat-docs/

[<img src="https://gist.githubusercontent.com/cxmeel/0dbc95191f239b631c3874f4ccf114e2/raw/documentation.svg" alt="Documentation" height="40" />](https://kenblu24.github.io/superneuromat-docs/)

Unlike its sister package, [SuperNeuroABM](https://github.com/kenblu24/superneuroabm), SuperNeuroMAT uses a matrix-based representation
of the network, which allows for more efficient simulation and GPU acceleration.

SuperNeuroMAT focuses on super-fast computation of Leaky Integrate and Fire **(LIF)** spiking neuron models with STDP.

It provides:
1. Support for leaky integrate and fire neuron model with the following parameters:
  * neuron threshold
  * neuron leak
  * neuron refractory period
2. Support for Spiking-Time-Dependent Plasticity (STDP) on synapses with:
  * weight
  * delay
  * per-synapse disabling of learning
3. Support for all-to-all connections as well as self connections
4. A turing-complete model of neuromorphic computing
5. Optional GPU acceleration or Optional Sparse computation

* Note that long delays may impact performance. Consider using an agent-based simulator
such as [SuperNeuroABM](https://github.com/ORNL/superneuroabm) for longer delays.


## Installation
1. Install using `pip install superneuromat`
2. Update/upgrade using `pip install superneuromat --upgrade`

The [installation guide](https://kenblu24.github.io/superneuromat-docs/guide/install.html)
covers virtual environments, faster installation with uv, installing support for CUDA GPU acceleration, and more.

## Usage
Import the spiking neural network class: 

```python
from superneuromat import SNN
```

See the [tutorial](https://kenblu24.github.io/superneuromat-docs/guide/firstrun.html) for more.

## Citation
1. Please cite SuperNeuroMAT using:
	```
	@inproceedings{date2023superneuro,
	  title={SuperNeuro: A fast and scalable simulator for neuromorphic computing},
	  author={Date, Prasanna and Gunaratne, Chathika and R. Kulkarni, Shruti and Patton, Robert and Coletti, Mark and Potok, Thomas},
	  booktitle={Proceedings of the 2023 International Conference on Neuromorphic Systems},
	  pages={1--4},
	  year={2023}
	}
	```
2. References for SuperNeuroMAT:
	- [SuperNeuro: A Fast and Scalable Simulator for Neuromorphic Computing](https://dl.acm.org/doi/abs/10.1145/3589737.3606000)
	- [Neuromorphic Computing is Turing-Complete](https://dl.acm.org/doi/abs/10.1145/3546790.3546806)
	- [Computational Complexity of Neuromorphic Algorithms](https://dl.acm.org/doi/abs/10.1145/3477145.3477154)
