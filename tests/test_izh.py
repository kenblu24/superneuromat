import unittest

import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat.izhmodel import IZHModel


mode = 'IB'

if mode == "IB":
    params = {
        'C': 150,
        'leak': 0.01,
        'b': 5,
        'refractory_period': 130,
        'refractory_state': 0,
        'k': 1.2,
        'vrest': -75,
        'reset_state': -56,
        'threshold': -45,
        # 'vpeak': 50,
        'bias': 420,
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


class LogicGatesTest(unittest.TestCase):
    """ Test SNNs for AND and OR gate

    """

    use = 'cpu'
    sparse = False

    def setUp(self):
        self.snn = IZHModel()
        self.snn.backend = self.use
        self.snn.sparse = self.sparse

    def test_izh(self):
        snn = self.snn

        # Create neurons
        a = snn.create_neuron(**params)

        # Setup and simulate
        v_hist = []
        for i in range(2000):
            v_hist.append(a.state)
            snn.simulate()

        # Print spike train and neuromorphic model
        snn.print_spike_train()
        print(snn)
        print(v_hist)

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


if __name__ == "__main__":
    unittest.main()
