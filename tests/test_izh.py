import unittest

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "../src/")

# from test_speed import RGNBenchmark
from superneuromat.izhmodel import IZHModel


mode = 'IB'

if mode == "IB":
    params = {
        'C': 150,
        'leak': 0.01,
        'b': 5,
        'initial_state': -75,
        'refractory_period': 130,
        'refractory_state': 0,
        'k': 1.2,
        'vrest': -75,
        'reset_state': -56,
        'threshold': -45,
        # 'vpeak': 50,
        # 'bias': 420,
        'bias': 304,
    }
elif mode == "CH":
    params = {
        'C': 50,
        'leak': 0.03,
        'b': 1,
        'refractory_period': 150,
        'refractory_state': 0,
        'k': 1.5,
        'vrest': -60,
        'reset_state': -40,
        'threshold': -40,
        # 'vpeak': 25,
        'bias': 190,
    }


class IZHTest(unittest.TestCase):
    """ Test SNNs for AND and OR gate

    """

    use = 'cpu'
    sparse = False

    def setUp(self):
        self.snn = IZHModel()
        self.snn.backend = self.use
        self.snn.sparse = self.sparse

    def get_rng(self, seed=None):
        if seed is None or isinstance(seed, int):
            return np.random.default_rng(seed)

    def make_randconn_net(self, size=100, sparsity=0.1, rng=None, neuron_params=None, synapse_params=None):
        rng = self.get_rng(rng)
        snn = IZHModel()

        neuron_params = {'threshold': 1} | (neuron_params or {})
        synapse_params = {'stdp_enabled': True} | (synapse_params or {})

        for _ in range(size):
            snn.create_neuron(**neuron_params)

        target = int(size * size * sparsity)
        selected_synapses = rng.binomial(1, sparsity, (size, size))
        selected_synapses = selected_synapses.astype(bool)
        delta = np.count_nonzero(selected_synapses) - target
        if delta > 0:
            candidates = np.array(selected_synapses.nonzero()).T
            selected_synapses[*rng.choice(candidates, size=(delta,), replace=False).T] = False
        elif delta < 0:
            candidates = np.array((~selected_synapses).nonzero()).T
            selected_synapses[*rng.choice(candidates, size=(abs(delta),), replace=False).T] = True
        assert np.count_nonzero(selected_synapses) == target

        for i, j in zip(*selected_synapses.nonzero()):
            snn.create_synapse(i, j, **synapse_params)

        return snn

    def add_spikes(self, snn: IZHModel, time_steps=10, rate=1, seed=None):
        rng = self.get_rng(seed)
        rate = min(rate, snn.num_neurons)
        for t in range(time_steps):
            for neuron in snn.neurons[rng.choice(snn.neurons.indices, rate, replace=False)]:
                neuron.add_spike(t)

    def test_izh(self):
        snn = self.snn

        # Create neurons
        a = snn.create_neuron(**params)

        # Setup and simulate
        v_hist = []
        for i in range(5000):
            v_hist.append(a.state)
            snn.simulate()

        # Print spike train and neuromorphic model
        snn.print_spike_train()
        print(snn)
        print(v_hist)
        fig, ax = plt.subplots()
        ax.plot(v_hist)
        plt.show()

        # expected_spike_train = [
        #     [0, 0, 0],  # in:  0┬0
        #     [0, 0, 0],  # out:  0
        #     [0, 1, 0],  # in:  0┬1
        #     [0, 0, 0],  # out:  0
        #     [1, 0, 0],  # in:  1┬0
        #     [0, 0, 0],  # out:  0
        #     [1, 1, 0],  # in:  1┬1
        #     [0, 0, 1],  # out:  1
        # ]
        # assert snn.ispikes.astype(int).tolist() == expected_spike_train

    def test_izh_two(self):
        snn = self.snn

        # Create neurons
        a = snn.create_neuron(**params)
        b = snn.create_neuron(**params)

        a.connect_child(b, weight=5000.0, delay=1)
        a.add_spikes([1000.0] * 10, exist='overwrite')

        # Setup and simulate
        states = []
        for i in range(2000):
            states.append(snn.neuron_states)
            snn.simulate()

        # Print spike train and neuromorphic model
        states = np.asarray(states)
        snn.print_spike_train()
        print(snn)
        # print(v_hist)
        fig, ax = plt.subplots()
        ax.plot(states)
        plt.show()

    def test_izh_big(self):
        self.snn = snn = self.make_randconn_net(size=20, sparsity=0.3, neuron_params=params)

        # Create neurons
        # a = snn.create_neuron(**params)

        # Setup and simulate
        states = []
        for i in range(1000):
            states.append(snn.neuron_states)
            snn.simulate()

        # Print spike train and neuromorphic model
        states = np.asarray(states)
        snn.print_spike_train()
        print(snn)
        # print(v_hist)
        fig, ax = plt.subplots()
        ax.plot(states)
        plt.show()


if __name__ == "__main__":
    unittest.main()
