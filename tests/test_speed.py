import unittest
import time
from io import StringIO

import gc

import numpy as np

import sys
sys.path.insert(0, "../src/")

from superneuromat import SNN


hparams = (
    "size\tsparsity\ttime_steps",
    "2	1.0	1",
    # "100	0.001	10",
    # "100	0.1	10",
    # "100	0.5	10",
    # "500	0.0001	10",
    # "500	0.001	10",
    "500	0.1	10",
    "500	0.5	10",
    # "1000	0.0001	10",
    # "1000	0.001	10",
    # "1000	0.01	10",
    # "1000	0.1	10",
    # "1000	0.15	10",
    # "1000	0.2	10",
    # "1000	0.25	10",
    # "1000	0.3	10",
    # "1000	0.0001	100",
    # "1000	0.001	100",
    # "1000	0.01	100",
    # "1000	0.1	100",
    # "1000	0.15	100",
    # "1000	0.2	100",
    # "1000	0.25	100",
    "1000	0.001	5",
    "1000	0.01	5",
    "1000	0.05	5",
    "1000	0.001	50",
    "1000	0.01	50",
    "1000	0.05	50",
    "1000	0.001	200",
    "1000	0.01	200",
    "1000	0.05	200",
    "1000	0.001	400",
    "1000	0.01	400",
    "1000	0.05	400",
    "2000	0.001	5",
    "2000	0.01	5",
    "2000	0.05	5",
    "2000	0.001	50",
    "2000	0.01	50",
    "2000	0.05	50",
    "2000	0.001	200",
    "2000	0.01	200",
    "2000	0.05	200",
    "2000	0.001	400",
    "2000	0.01	400",
    "2000	0.05	400",
    # "2000	0.3	100",
    "3000	0.001	5",
    "3000	0.01	5",
    "3000	0.05	5",
    "3000	0.001	50",
    "3000	0.01	50",
    "3000	0.05	50",
    "3000	0.001	200",
    "3000	0.01	200",
    "3000	0.05	200",
    "3000	0.001	400",
    "3000	0.01	400",
    "3000	0.05	400",
    # "3000	0.3	100",
    # "4000	0.3	100",
    # "4000	0.001	5",
    # "4000	0.01	5",
    # "4000	0.05	5",
    # "4000	0.001	50",
    # "4000	0.01	50",
    # "4000	0.05	50",
    # "4000	0.001	200",
    # "4000	0.01	200",
    # "4000	0.05	200",
    # "4000	0.001	400",
    # "4000	0.01	400",
    # "4000	0.05	400",
    # "5000	0.2	100",
    # "5000	0.2	50",
    # "5000	0.2	200",
    # "5000	0.2	300",
    # "5000	0.2	400",
    # "5000	0.2	500",
    # "6000	0.3	100",
    # "6000	0.2	200",
    # "6000	0.2	300",
    # "6000	0.2	500",
    # "6000	0.001	5",
    # "6000	0.01	5",
    # "6000	0.05	5",
    # "6000	0.001	50",
    # "6000	0.01	50",
    # "6000	0.05	50",
    # "6000	0.001	200",
    # "6000	0.01	200",
    # "6000	0.05	200",
    # "6000	0.001	400",
    # "6000	0.01	400",
    # "6000	0.05	400",
    # "8000	0.2	2",
    # "8000	0.2	5",
    # "8000	0.2	10",
    # "8000	0.2	20",
    # "8000	0.2	50",
    # "8000	0.2	80",
    # "8000	0.2	100",
    # "8000	0.2	150",
    # "8000	0.2	200",
    # "8000	0.1	200",
    # "8000	0.001	5",
    # "8000	0.01	5",
    # "8000	0.05	5",
    # "8000	0.001	50",
    # "8000	0.01	50",
    # "8000	0.05	50",
    # "8000	0.001	200",
    # "8000	0.01	200",
    # "8000	0.05	200",
    # "8000	0.001	400",
    # "8000	0.01	400",
    # "8000	0.05	400",
    # "8000	0.1	300",
    # "8000	0.1	400",
    # "8000	0.15	100",
    # "8000	0.15	200",
    # "8000	0.15	300",
    # "8000	0.2	300",
    # "8000	0.2	500",
    # "10000	0.001	100",
    # "10000	0.01	100",
    # "10000	0.1	100",
    # "10000	0.5	100",
    # "10000	0.001	1000",
    # "10000	0.01	1000",
    # "10000	0.1	1000",
    # "10000	0.5	1000",
)
hparams = '\n'.join(hparams)


class RGNBenchmark(unittest.TestCase):
    """ Test sparse operations

    """

    def get_rng(self, seed=None):
        if seed is None or isinstance(seed, int):
            return np.random.default_rng(seed)

    def make_randconn_net(self, size=100, sparsity=0.1, rng=None, neuron_params=None, synapse_params=None):
        rng = self.get_rng(rng)
        snn = SNN()

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

    def add_spikes(self, snn: SNN, time_steps=10, rate=1, seed=None):
        rng = self.get_rng(seed)
        rate = min(rate, snn.num_neurons)
        for t in range(time_steps):
            for neuron in snn.neurons[rng.choice(snn.neurons.indices, rate, replace=False)]:
                neuron.add_spike(t)

    def timesim(self, snn: SNN, time_steps=10):
        start = time.time()
        snn.simulate(time_steps)
        snn.release_mem()
        ret = time.time() - start
        del snn
        gc.collect()
        return ret

    def benchmark_(self, snn: SNN, time_steps, backend, sparse, avg=1):
        results = []

        for _ in range(avg):
            net = snn.copy()
            net.backend = backend
            net.sparse = sparse
            results.append(self.timesim(net, time_steps))

        return np.mean(results)

    def benchmark_backends(self, size=100, sparsity=0.1, backends=('cpu-sparse', 'cpu-dense', 'jit', 'gpu'),
                           time_steps=10, seed=42, neuron_params=None, synapse_params=None, avg=1):
        snn = self.make_randconn_net(size, sparsity, rng=seed, neuron_params=neuron_params, synapse_params=synapse_params)
        self.add_spikes(snn, time_steps=time_steps, rate=10, seed=seed)
        data = {}
        if 'cpu-dense' in backends:
            data['cpu-dense'] = self.benchmark_(snn, time_steps, 'cpu', sparse=False, avg=avg)
        if 'cpu-sparse' in backends:
            data['cpu-sparse'] = self.benchmark_(snn, time_steps, 'cpu', sparse=True, avg=avg)
        if 'jit' in backends:
            data['jit'] = self.benchmark_(snn, time_steps, 'jit', sparse=False, avg=avg)
        if 'gpu' in backends:
            data['gpu'] = self.benchmark_(snn, time_steps, 'gpu', sparse=False, avg=avg)
        del snn
        gc.collect()

        return data

    def test_test(self):
        import pandas as pd
        params = pd.read_csv(StringIO(hparams), sep='\t')
        data = []
        for _i, size, sparsity, time_steps in params.itertuples():
            data.append(self.benchmark_backends(size, sparsity, time_steps=time_steps, avg=5))
            d = data[-1]
            print(f"size: {size}, sparsity: {sparsity}, time_steps: {time_steps}, cpu-dense: {d['cpu-dense']:.3f}, cpu-sparse: {d['cpu-sparse']:.3f}, jit: {d['jit']:.3f}, gpu: {d['gpu']:.3f}")
        data = pd.concat([params, pd.DataFrame(data)], axis=1)
        print(data)
        data.to_csv('superneuromat_benchmark_results.csv')


if __name__ == "__main__":
    unittest.main()
