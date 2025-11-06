import warnings

import numpy as np
from scipy.sparse import csc_array

from .neuromorphicmodel import SNN
from .accessor_classes import Neuron
from .util import float_err, int_err, is_intlike_catch


# typing
from .accessor_classes import Synapse
from typing import Any


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

        self.isyn = []  # single exponential decay synapse
        self.isyn_alpha = []  # double exponential decay synapse

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

    def create_synapse(
        self,
        pre_id: int | Neuron,
        post_id: int | Neuron,
        weight: float = 1.0,
        delay: int = 1,
        stdp_enabled: bool | Any = False,
        exist: str = "error",
        **kwargs,
    ) -> Synapse:
        """Creates a synapse in the SNN

        Creates synapse connecting a pre-synaptic neuron to a post-synaptic neuron
        with a given set of synaptic parameters (weight, delay and stdp_enabled)

        Parameters
        ----------
        pre_id : int | Neuron
            ID of the pre-synaptic neuron (spike sender).
        post_id : int | Neuron
            ID of the post-synaptic neuron (spike destination).
        weight : float, default=1.0
            Synaptic weight; weight is multiplied to the incoming spike.
        delay : int, default=1
            Synaptic delay; number of time steps by which the outgoing signal of the synapse is delayed by.
        stdp_enabled : bool | Any, default=False
            If True, stdp will be enabled on the synapse, allowing the weight of this synapse to be updated.
        exist : str, default='error'
            Action if synapse  already exists with the exact pre- and post-synaptic neurons.
            Should be one of ['error', 'overwrite', 'dontadd'].


        If a delay is specified, a chain of neurons and synapses will automatically be added to the model
        to represent the delay, and this function will return the last synapse of the chain.
        The other neurons and synapses in the chain can be accessed via the :py:attr:`delay_chain` and
        :py:attr:`delay_chain_synapses` properties of the synapse, respectively.

        While only positive delay values are supported due to temporal consistency and causality requirements,
        The delay will be stored as ``delay * -1`` in the model to represent that it is a chained delay.
        This does not affect the effective delay value, as the delay will still be applied via the delay chain.

        Note that delays of delay chains cannot be modified after creation.

        Raises
        ------
        TypeError

            * ``pre_id`` or ``post_id`` is not neuron or neuron ID (``int``).
            * ``weight`` is not a ``float``.
            * ``delay`` cannot be cast to ``int``.
            * ``exist`` is not a ``str``.

        ValueError

            * ``pre_id`` or ``post_id`` is not a valid neuron or neuron ID.
            * ``delay`` is less than or equal to ``0``
            * ``exist`` is not one of ``'error', 'overwrite', 'dontadd'``.
            * Synapse with the given pre- and post-synaptic neurons already exists, ``exist='overwrite'``, and ``delay != 1``.

        RuntimeError

            * Synapse with the given pre- and post-synaptic neurons already exists and ``exist='error'``.

        Returns
        -------
        Synapse


        .. seealso::

           :py:meth:`Neuron.connect_child`, :py:meth:`Neuron.connect_parent`
        """

        # TODO: make delay chaining an SNN option
        # TODO: ensure created hidden synapses are not flagged as newdelay

        # Ensure we work with neuron ids
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if isinstance(post_id, Neuron):
            post_id = post_id.idx

        # input validation
        fname = 'create_synapse()'

        if not is_intlike_catch(pre_id):
            raise TypeError("pre_id must be int or Neuron.")
        pre_id = int(pre_id)

        if not is_intlike_catch(post_id):
            raise TypeError("post_id must be int or Neuron.")
        post_id = int(post_id)

        weight = float_err(weight, 'weight', fname)
        delay = int_err(delay, 'delay', fname)

        if pre_id < 0:
            raise ValueError("pre_id must be greater than or equal to zero")
        elif not pre_id < self.num_neurons:
            msg = f"Added synapse to non-existent pre-synaptic Neuron {pre_id}."
            raise warnings.warn(msg, stacklevel=2)

        if post_id < 0:
            raise ValueError("post_id must be greater than or equal to zero")
        if not post_id < self.num_neurons:
            msg = f"Added synapse to non-existent post-synaptic Neuron {post_id}."
            raise warnings.warn(msg, stacklevel=2)

        if (enable_stdp := kwargs.pop('enable_stdp', None)) is not None:
            warnings.warn("create_synapse kwarg 'enable_stdp' is deprecated. Use 'stdp_enabled' instead.",
                          DeprecationWarning, stacklevel=2)
            stdp_enabled = enable_stdp

        ambiguous = ('true', '1', 'y', 'yes', 'on', 'f', 'false', '0', 'n', 'no', 'off')
        if isinstance(stdp_enabled, str) and stdp_enabled.lower() in ambiguous:
            msg = f"{fname} argument stdp_enabled received {stdp_enabled!r}"
            msg += " which has ambiguous truthiness. Consider using an explicit boolean value instead."
            warnings.warn(msg, stacklevel=2)

        last_in_chain = kwargs.pop('_is_last_chained_synapse', False)
        if delay <= 0 and not last_in_chain:
            raise ValueError("delay must be greater than or equal to 1")

        if kwargs:
            msg = f"create_synapse() received unexpected keyword arguments: {list(kwargs.keys())}"
            raise TypeError(msg)

        if (idx := self.get_synapse_id(pre_id, post_id)) is not None:  # if synapse already exists
            if not isinstance(exist, str):
                raise TypeError("exist must be a string")
            exist = exist.lower()
            if exist == "error":
                msg = f"Synapse already exists: {self.synapses[idx]!s}"
                msg += "If this was intentional, choose arg exist=<'dontadd', 'overwrite'>."
                raise RuntimeError(msg)
            elif exist == "overwrite":
                # check if delay has changed
                if delay != self.synaptic_delays[idx]:
                    raise ValueError("create_synapse() tried to overwrite chained synapse with different delay.")
                # overwrite old synapse params
                self.pre_synaptic_neuron_ids[idx] = pre_id
                self.post_synaptic_neuron_ids[idx] = post_id
                self.synaptic_weights[idx] = weight
                self.synaptic_delays[idx] = delay
                self.enable_stdp[idx] = stdp_enabled
            elif exist == "dontadd":
                return self.synapses[idx]
            else:
                msg = f"Invalid value for exist: {exist}. Expected 'error', 'overwrite', or 'dontadd'."
                raise ValueError(msg)
            return self.synapses[idx]  # prevent fall-through if user catches the error

        # Set new synapse parameters
        if delay == 1 or last_in_chain:
            self.pre_synaptic_neuron_ids.append(pre_id)
            self.post_synaptic_neuron_ids.append(post_id)
            self.synaptic_weights.append(weight)
            self.synaptic_delays.append(delay)
            self.enable_stdp.append(stdp_enabled)
            self.connection_ids[(pre_id, post_id)] = self.num_synapses - 1
        else:
            for _d in range(int(delay) - 1):  # delay by stringing together hidden synapses
                temp_id = self.create_neuron()
                self.create_synapse(pre_id, temp_id)
                pre_id = temp_id
            # place weight on last hidden synapse
            self.create_synapse(pre_id, post_id, weight=weight, stdp_enabled=stdp_enabled,
                                delay=-delay, _is_last_chained_synapse=True)  # , chained_neuron_delay=True)

        # Return synapse ID
        return Synapse(self, self.num_synapses - 1)

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
