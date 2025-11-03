import numpy as np
from scipy.sparse import csc_array

from .neuromorphicmodel import SNN
from .accessor_classes import Neuron
from .util import float_err, int_err


class IZHModel(SNN):
    """Spiking Neural Network model implementing IZH and STDP using matrix representations."""

    def __init__(self):
        """Initialize the IZHModel"""
        super().__init__()

        self.C = []  # capacitance
        # self.a = self.neuron_leaks  # time constant for u
        self.b = []  # feedback
        # self.d = self.neuron_refractory_periods
        self.k = []  # defines shape of nullcline/iv curve
        # self.u = self.neuron_refractory_periods_state
        self.vrest = []
        self.vpeak = []
        self.bias = []
        self.dt = 0.1

    def create_neuron(
        self,
        threshold: float = 0.0,
        leak: float = np.inf,
        reset_state: float = 0.0,
        refractory_period: int = 0,
        refractory_state: int = 0,
        initial_state: float | None = -60.0,
        C: float = 1.0,
        b: float = 0.0,
        k: float = 1.0,
        vrest: float = -60.0,
        vpeak: float = 30.0,
        bias: float = 0.0,
    ) -> Neuron:
        """
        Create a neuron in the SNN.

        Parameters
        ----------
        threshold : float, default=0.0
            Neuron threshold; the neuron spikes if its internal state is strictly
            greater than the neuron threshold
        leak : float, default=numpy.inf
            Neuron leak; the amount by which the internal state of the neuron is
            pushed towards its reset state
        reset_state : float, default=0.0
            Reset state of the neuron; the value assigned to the internal state
            of the neuron after spiking
        refractory_period : int, default=0
            Refractory period of the neuron; the number of time steps for which
            the neuron remains in a dormant state after spiking

        Returns
        -------
        Neuron
            The :py:class:`Neuron` object.

        Raises
        ------
        TypeError
            If `threshold`, `leak`, or `reset_state` is not a float or int, or if
            `refractory_period` is not an int.
        ValueError
            If `leak` is less than 0.0 or `refractory_period` is less than 0.


        .. hint::

           :py:attr:`Neuron.idx` is the ID of the created neuron.

        """
        # Input validation
        fname = 'create_neuron()'

        leak = float_err(leak, 'leak', fname)
        if not self.allow_signed_leak and leak < 0.0:
            raise ValueError("leak must be greater than or equal to zero.")

        C = float_err(C, 'capacitance (C)', fname)
        if C <= 0.0:
            raise ValueError("capacitance (C) must be greater than or equal to zero.")

        b = float_err(b, 'feedback (b)', fname)
        k = float_err(k, 'shape of nullcline/iv curve (k)', fname)
        vrest = float_err(vrest, 'reset voltage (vrest)', fname)
        bias = float_err(bias, 'bias (Ib)', fname)

        refractory_period = int_err(refractory_period, 'refractory_period', fname)
        if refractory_period < 0:
            raise ValueError("refractory_period must be greater than or equal to zero.")

        refractory_state = int_err(refractory_state, 'refractory_state', fname)
        if refractory_state < 0:
            raise ValueError("refractory_state must be greater than or equal to zero.")

        # Add neurons to SNN
        self.neuron_thresholds.append(float_err(threshold, 'threshold', fname))
        self.neuron_leaks.append(leak)
        self.neuron_reset_states.append(float_err(reset_state, 'reset_state', fname))
        self.neuron_refractory_periods.append(refractory_period)
        self.neuron_refractory_periods_state.append(refractory_state)
        self.neuron_states.append(reset_state if initial_state is None else float_err(initial_state, 'initial_state', fname))
        self.C.append(C)
        self.b.append(b)
        self.k.append(k)
        self.vrest.append(vrest)
        self.vpeak.append(vpeak)
        self.bias.append(bias)

        # Return neuron ID
        return Neuron(self, self.num_neurons - 1)

    def _setup(self, dtype=None, sparse=None):
        """Setup the SNN for simulation. Not intended to be called by end user."""
        super()._setup(dtype=dtype, sparse=sparse)

        self._C = np.asarray(self.C, self.dd)
        # self._a = np.asarray(self.a, self.dd)
        self._b = np.asarray(self.b, self.dd)
        # self._d = np.asarray(self.d, self.dd)
        self._k = np.asarray(self.k, self.dd)
        # self._u = np.asarray(self.u, self.dd)
        self._vpeak = np.asarray(self.vpeak, self.dd)
        self._vrest = np.asarray(self.vrest, self.dd)
        self._I = np.asarray(self.bias, self.dd)

    def simulate_cpu(self, time_steps: int = 1000, callback=None) -> None:
        self._last_used_backend = 'cpu'

        if self._do_stdp:
            if not self._do_positive_update:
                self._stdp_Apos = np.zeros(len(self._stdp_Aneg), self.dd)
            if not self._do_negative_update:
                self._stdp_Aneg = np.zeros(len(self._stdp_Apos), self.dd)
            self._Aneg = np.array(self._stdp_Aneg[::-1])
            self._Asum = (np.asarray(self._stdp_Apos[::-1]) - np.asarray(self._Aneg)
                          ).reshape((-1, 1))

        v = self._internal_states
        u = self._neuron_refractory_periods
        d = self._neuron_refractory_periods_original
        a = self._neuron_leaks
        b = self._b
        k = self._k
        vrest = self._vrest
        vthr = self._neuron_thresholds
        vpeak = self._vpeak
        Ibias = self._I

        # Simulate
        for tick in range(time_steps):
            if callback is not None:
                if callable(callback):
                    callback(self, tick, time_steps)

            dv = (
                    k * (v - vrest) * (v - vthr) - u + Ibias
                    + self._input_spikes[tick] + (self._weights.T @ self._spikes)
                ) / self._C

            v += self.dt * dv
            u += self.dt * (a * (b * (v - vrest) - u))

            # Compute spikes
            self._spikes = np.greater(v, vpeak).astype(self.dbin)

            u += d * self._spikes

            # Refractory period: Compute indices of neuron which are in their refractory period
            # indices = np.greater(self._neuron_refractory_periods, 0)

            # For neurons in their refractory period, zero out their spikes and decrement refractory period by one
            # self._spikes[indices] = 0
            # self._neuron_refractory_periods[indices] -= 1

            # For spiking neurons, turn on refractory period
            mask = self._spikes.astype(bool)
            v[mask] = self._neuron_reset_states[mask]
            # self._neuron_refractory_periods[mask] = self._neuron_refractory_periods_original[mask]

            # Reset internal states
            # self._internal_states[mask] = self._neuron_reset_states[mask]

            # Append spike train
            self.spike_train.append(self._spikes)

            # STDP Operations
            t = min(self.stdp_time_steps, len(self.spike_train) - 1)

            if self._do_stdp and t > 0:
                if self._is_sparse:
                    Sprev = csc_array(self.spike_train[-t - 1:-1],
                                      shape=[t, self.num_neurons], dtype=self.dd)
                    Scurr = csc_array([self.spike_train[-1]] * t,
                                      shape=[t, self.num_neurons], dtype=self.dd)
                else:
                    Sprev = np.asarray(self.spike_train[-t - 1:-1], dtype=self.dd)
                    Scurr = np.asarray([self.spike_train[-1]] * t, dtype=self.dd)

                self._weights += ((((self._Asum[-t:] * Sprev).T @ Scurr) * self._stdp_enabled_synapses)
                                + (self._Aneg[-t:].sum() * self._stdp_enabled_synapses))
                if self._is_sparse:
                    self._weights = self._weights.astype(self.dd)

        if not self.manual_setup:
            self.devec()
            self.consume_input_spikes(time_steps)
