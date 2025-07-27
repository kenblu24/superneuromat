import numpy as np
from numba import jit


@jit(nopython=True)
def apply_leak_jit(states, reset_states, leaks):
    # CAUTION: This function has side-effects (not a pure function)

    # Leak: internal state > reset state
    indices = states > reset_states
    states[indices] = np.maximum(
        states[indices] - leaks[indices], reset_states[indices]
    )

    # Leak: internal state < reset state
    indices = states < reset_states
    states[indices] = np.minimum(
        states[indices] + leaks[indices], reset_states[indices]
    )


@jit(nopython=True)
def lif_jit(
    tick: int,
    input_spikes,
    spikes,
    states,
    thresholds,
    leaks,
    reset_states,
    refractory_periods,
    refractory_periods_original,
    weights,
    delays: None,
    delayed_synapses: None,
    synaptic_delaysT: None,
    delayed_spikes: None,
):
    # CAUTION: This function has side-effects (not a pure function)
    # spikes and states and refractory_periods are modified in-place
    # ______     ______     __________________
    # DO NOT ASSIGN THESE VARIABLES WITHIN THIS FUNCTION or things will break
    # DO NOT states = something

    apply_leak_jit(states, reset_states, leaks)

    # Internal state (in-place)
    states += input_spikes[tick] + (weights.T @ spikes)

    # Compute spikes (in-place) into spikes (numba doesn't support keyword 'out')
    np.greater(states, thresholds, spikes)

    # Refractory period: Compute indices of neuron which are in their refractory period
    indices = refractory_periods > 0

    # For neurons in their refractory period, zero out their spikes and decrement refractory period by one
    spikes[indices] = 0
    refractory_periods[indices] -= 1

    # For spiking neurons, turn on refractory period
    mask = spikes.astype(np.bool_)
    refractory_periods[mask] = refractory_periods_original[mask]

    # Reset internal states (in-place)
    states[mask] = reset_states[mask]

    # states, spikes, refractory_periods were modified in-place
    # everything else is local


@jit(nopython=True)
def lif_jit_undelayed(
    tick: int,
    input_spikes,
    spikes,
    states,
    thresholds,
    leaks,
    reset_states,
    refractory_periods,
    refractory_periods_original,
    weights,
    undelayed_weightsT,  # empty buffer that we fill
    delayed_synapses,
    delayed_spikes,
):
    # CAUTION: This function has side-effects (not a pure function)

    apply_leak_jit(states, reset_states, leaks)

    weightsT = weights.T
    synapse_mask = ~delayed_synapses.T
    undelayed_weightsT = weightsT[synapse_mask]

    # add to internal state of neurons which are not delayed
    states += input_spikes[tick] + (undelayed_weightsT @ spikes)

    undelayed_weightsT[synapse_mask] = 0.0

    states += delayed_spikes

    np.greater(states, thresholds, spikes)

    indices = refractory_periods > 0

    spikes[indices] = 0
    refractory_periods[indices] -= 1

    mask = spikes.astype(np.bool_)
    refractory_periods[mask] = refractory_periods_original[mask]

    states[mask] = reset_states[mask]

    # states, spikes, refractory_periods were modified in-place
    # everything else is local


@jit(nopython=True)
def lif_jit_delayed(
    delay: int,
    spikes,
    states,
    weights,
    delayed_weightsT,  # empty buffer that we fill
    delayed_synapses: dict,
    synaptic_delaysT,
):
    # TODO: setup translation of delayed_synapses to a numba typed dict
    pass


@jit(nopython=True)
def stdp_update_jit(tsteps, spike_train, weights_pointer, apos, aneg, stdp_enabled):
    # STDP Operations
    for i in range(tsteps):
        update_synapses = np.outer(spike_train[~i - 1], spike_train[-1])
        if apos[i]:
            weights_pointer += apos[i] * update_synapses * stdp_enabled
        if aneg[i]:
            weights_pointer += aneg[i] * (1 - update_synapses) * stdp_enabled


@jit(nopython=True)
def stdp_update_jit_apos(tsteps, spike_train, weights_pointer, apos, stdp_enabled):
    # STDP Operations
    for i in range(tsteps):
        if apos[i]:
            update_synapses = np.outer(spike_train[~i - 1], spike_train[-1])
            weights_pointer += apos[i] * update_synapses * stdp_enabled


@jit(nopython=True)
def stdp_update_jit_aneg(tsteps, spike_train, weights_pointer, aneg, stdp_enabled):
    # STDP Operations
    for i in range(tsteps):
        if aneg[i]:
            update_synapses = np.outer(spike_train[~i - 1], spike_train[-1])
            weights_pointer += aneg[i] * (1 - update_synapses) * stdp_enabled
