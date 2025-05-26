import unittest
import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


class RefractoryTest(unittest.TestCase):
    """ Test refractory period

    """

    use = 'cpu'
    sparse = False

    def setUp(self):
        self.snn = SNN()
        self.snn.backend = self.use
        self.snn.sparse = self.sparse

    def test_refractory_one(self):
        print("One neuron refractory period test")

        snn = self.snn

        n_id = snn.create_neuron(refractory_period=2).idx

        snn.add_spike(1, n_id, 1)
        snn.add_spike(2, n_id, 3)
        snn.add_spike(3, n_id, 4)
        snn.add_spike(4, n_id, 1)

        snn.simulate(10)

        snn.print_spike_train()
        print()

        expected_spike_train = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        expected_spike_train = np.reshape(expected_spike_train, (-1, 1)).tolist()
        assert snn.ispikes.tolist() == expected_spike_train

    def test_refractory_two(self):
        print("Two neuron refractory period test")

        snn = self.snn

        n1 = snn.create_neuron(threshold=-1.0, reset_state=-1.0, refractory_period=2)
        n2 = snn.create_neuron(refractory_period=1000000)

        snn.create_synapse(n1, n2, weight=2.0, delay=2, use_chained_delay=True)

        snn.add_spike(1, n2, -1.0)
        snn.add_spike(2, n1, 10.0)
        snn.add_spike(3, n1, 10.0)
        snn.add_spike(5, n1, 10.0)

        snn.simulate(10)

        snn.print_spike_train()

        expected_spike_train = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        assert snn.ispikes.tolist() == expected_spike_train

    def test_refractory_newdelay(self):
        print("refractory new delay test")

        snn = self.snn
        snn.use_chained_delay = False

        n1 = snn.create_neuron(threshold=-1.0, reset_state=-1.0, refractory_period=2)
        n2 = snn.create_neuron(refractory_period=1000000)

        snn.create_synapse(n1, n2, weight=2.0, delay=2)

        snn.add_spike(1, n2, -1.0)
        snn.add_spike(2, n1, 10.0)
        snn.add_spike(3, n1, 10.0)
        snn.add_spike(5, n1, 10.0)

        snn.simulate(10)

        snn.print_spike_train()

        expected_spike_train = [
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
        assert snn.ispikes.tolist() == expected_spike_train

    def test_refractory_newdelays(self):
        print("refractory more delay test (from test_display)")
        snn = self.snn

        n0 = snn.create_neuron(threshold=-1.0, leak=2.0, refractory_period=3, reset_state=-2.0)
        n1 = snn.create_neuron(threshold=0.0, leak=1.0, refractory_period=1, reset_state=-2.0)
        n2 = snn.create_neuron(threshold=2.0, leak=0.0, refractory_period=0, reset_state=-1.0)
        n3 = snn.create_neuron(threshold=5.0, leak=np.inf, refractory_period=2, reset_state=-2.0)
        n4 = snn.create_neuron(threshold=-2.0, leak=5.0, refractory_period=1, reset_state=-2.0)

        chain = False

        snn.create_synapse(n0, n1)
        snn.create_synapse(n0, n2)
        snn.create_synapse(n0, n3, weight=4.0, delay=3, stdp_enabled=True, use_chained_delay=chain)
        snn.create_synapse(n4, n2, weight=2.0, delay=2, stdp_enabled=False, use_chained_delay=chain)
        syn = snn.create_synapse(n2, n1, weight=30.0, delay=4, stdp_enabled=True, use_chained_delay=chain)

        snn.add_spike(0, n2, 4.0)
        snn.add_spike(1, n1, 3.0)
        snn.add_spike(0, n3, 2.0)
        snn.add_spike(15, n3, 7.1)
        snn.add_spike(16, n1, 2.1)
        snn.add_spike(20, n4, 2.1)

        print(snn)
        print(bin(syn.flags()))

        # history = []
        for _i in range(21):
            snn.simulate(1)
            # history.append(snn.weight_mat())
            # print(_i)
            # print(snn.neuron_info())
            # print(snn.delayed_spikes)

        # for i, array in enumerate(history):
        #     for j, line in enumerate(str(array).split('\n')):
        #         if j == 0:
        #             print(f"{i:> 2d}: {line}")
        #         else:
        #             print(f"     {line}")


        expected_spikes = [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1]
        ]

        snn.print_spike_train()

        assert np.array_equal(snn.ispikes, expected_spikes)
        assert (snn.ispikes.sum() == np.sum(expected_spikes))


if __name__ == "__main__":
    unittest.main()
