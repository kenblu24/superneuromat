import math
import copy
import warnings
import numpy as np
from .accessor_classes import Neuron, Synapse, NeuronList, SynapseList

from typing import Any

try:
    import numba
    from numba import cuda
    from .numba_jit import lif_jit, stdp_update_jit
    from .numba_jit import stdp_update_jit_apos, stdp_update_jit_aneg
except ImportError:
    numba = None

try:
    from scipy.sparse import csc_array
except ImportError:
    csc_array = None


"""

FEATURE REQUESTS:

1. Visualize spike raster
2. Monitor STDP synapses

"""


def check_numba():
    msg = """Numba is not installed. Please install Numba to use this feature.
    You can install JIT support for SuperNeuroMAT with `pip install superneuromat[jit]`,
    or install Numba manually with `pip install numba`.
    """
    global numba
    if numba is None:
        try:
            import numba
        except ImportError as err:
            raise ImportError(msg) from err


def check_gpu():
    msg = """GPU support is not installed. Please install Numba to use this feature.
    You can install JIT support for SuperNeuroMAT with `pip install superneuromat[gpu]`,
    or see the install instructions for GPU support at https://ornl.github.io/superneuromat/guide/install.html#gpu-support.
    """
    global numba
    if numba is None:
        try:
            from numba import cuda
        except ImportError as err:
            raise ImportError(msg) from err


def is_intlike(x):
    if isinstance(x, int):
        return True
    else:
        return x == int(x)


def resize_vec(a, len, dtype=np.float64):
    a = np.asarray(a, dtype=dtype)
    if len(a) < len:
        return a[:len].copy()
    elif len(a) > len:
        return np.pad(a, (0, len - len(a)), 'constant')
    else:
        return a


class NeuromorphicModel:
    """Defines a neuromorphic model with neurons and synapses

    Parameters
    ----------
    num_neurons : int
        Number of neurons in the neuromorphic model
    neuron_thresholds : list
        List of neuron thresholds
    neuron_leaks : list
        List of neuron leaks
        defined as the amount by which the internal states of the neurons are pushed towards the neurons' reset states
    neuron_reset_states : list
        List of neuron reset states
    neuron_refractory_periods : list
        List of neuron refractory periods
    num_synapses : int
        Number of synapses in the neuromorphic model
    pre_synaptic_neuron_ids : list
        List of pre-synaptic neuron IDs
    post_synaptic_neuron_ids : list
        List of post-synaptic neuron IDs
    synaptic_weights : list
        List of synaptic weights
    synaptic_delays : list
        List of synaptic delays
    enable_stdp : list
        List of Boolean values denoting whether STDP learning is enabled on each synapse
    input_spikes : dict
        Dictionary of input spikes indexed by time
    spike_train : list
        List of spike trains for each time step
    stdp : bool
        Boolean parameter that denotes whether STDP learning has been enabled in the neuromorphic model
    stdp_time_steps : int
        Number of time steps over which STDP updates are made
    stdp_Apos : list
        List of STDP parameters per time step for excitatory update of weights
    stdp_Aneg : list
        List of STDP parameters per time step for inhibitory update of weights

    Methods
    -------
    create_neuron: Creates a neuron in the neuromorphic model
    create_synapse: Creates a synapse in the neuromorphic model
    add_spike: Add an external spike at a particular time step for a given neuron with a given value
    stdp_setup: Setup the STDP parameters
    setup: Setup the neuromorphic model and prepare for simulation
    simulate: Simulate the neuromorphic model for a given number of time steps
    print_spike_train: Print the spike train


    .. warning::

            1. Delay is implemented by adding a chain of proxy neurons.

                A delay of 10 between neuron A and neuron B would add 9 proxy neurons between A and B.

            2. Leak brings the internal state of the neuron closer to the reset state.

                The leak value is the amount by which the internal state of the neuron is pushed towards its reset state.

            3. Deletion of neurons may result in unexpected behavior.

            4. Input spikes can have a value

            5. All neurons are monitored by default

    """

    gpu_threshold = 10000
    jit_threshold = 1000
    disable_performance_warnings = True

    def __init__(self):
        """Initialize the neuromorphic model"""

        # Neuron parameters
        self.neuron_thresholds = []
        self.neuron_leaks = []
        self.neuron_states = []
        self.neuron_reset_states = []
        self.neuron_refractory_periods = []
        self.neuron_refractory_periods_state = []

        # Synapse parameters
        # self.num_synapses = 0
        self.pre_synaptic_neuron_ids = []
        self.post_synaptic_neuron_ids = []
        self.synaptic_weights = []
        self.synaptic_delays = []
        self.enable_stdp = []

        # Input spikes (can have a value)
        self.input_spikes = {}

        # Spike trains (monitoring all neurons)
        self.spike_train = []

        # STDP Parameters
        self.stdp = True
        self._stdp_Apos = []
        self._stdp_Aneg = []
        self._do_stdp = False
        self.stdp_positive_update = True
        self.stdp_negative_update = True

        self.default_dtype = np.float64

        self.neurons = NeuronList(self)
        self.synapses = SynapseList(self)

        self.gpu = numba and cuda.is_available()
        self._backend = 'auto'  # default backend setting. can be overridden at simulate time.
        self._last_used_backend = None
        self._sparse = False  # default sparsity setting.
        self._return_sparse = False  # whether to return spikes sparsely
        self._use_sparse = False
        self.manual_setup = False

    def last_used_backend_type(self):
        return self._last_used_backend

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value):
        # TODO: validate if selected backend is available and valid
        self._backend = value

    @property
    def dd(self):
        return self.default_dtype

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, sparsity: bool | Any):
        # TODO: validate if sparsity is available i.e. does user have scipy
        self._sparse = sparsity

    @property
    def num_neurons(self):
        return len(self.neuron_thresholds)

    @property
    def num_synapses(self):
        return len(self.pre_synaptic_neuron_ids)

    @property
    def stdp_time_steps(self):
        if self.stdp_positive_update and self.stdp_negative_update:
            assert len(self._stdp_Apos) == len(self._stdp_Aneg)  # pyright:ignore[reportArgumentType]
        if self.stdp_positive_update:
            return len(self._stdp_Apos)  # pyright:ignore[reportArgumentType]
        elif self.stdp_negative_update:
            return len(self._stdp_Aneg)  # pyright:ignore[reportArgumentType]
        else:
            return 0

    @property
    def neuron_df(self):
        """Returns a DataFrame containing information about neurons."""
        import pandas as pd
        return pd.DataFrame({
            "Neuron ID": list(range(self.num_neurons)),
            "Threshold": self.neuron_thresholds,
            "Leak": self.neuron_leaks,
            "Reset State": self.neuron_reset_states,
            "Refractory Period": self.neuron_refractory_periods,
        })

    @property
    def synapse_df(self):
        """Returns a DataFrame containing information about synapses."""
        import pandas as pd
        return pd.DataFrame({
            "Pre Neuron ID": self.pre_synaptic_neuron_ids,
            "Post Neuron ID": self.post_synaptic_neuron_ids,
            "Weight": self.synaptic_weights,
            "Delay": self.synaptic_delays,
            "STDP Enabled": self.enable_stdp,
        })

    @property
    def stdp_info(self):
        """Returns a string containing information about STDP."""
        return (
            f"STDP Enabled: {self.stdp} \n"
            + f"STDP Time Steps: {self.stdp_time_steps} \n"
            + f"STDP A positive: {self._stdp_Apos} \n"
            + f"STDP A negative: {self._stdp_Aneg}"
        )

    @property
    def ispikes(self):
        return np.asarray(self.spike_train, dtype=bool)

    def neuron_spike_totals(self, time_index=None):
        # TODO: make time_index reference global/model time, not just ispikes index, which may have been cleared
        if time_index is None:
            return np.sum(self.ispikes, axis=0)
        else:
            return np.sum(self.ispikes[time_index], axis=0)

    def __str__(self):
        return self.prettys()

    def prettys(self):
        import pandas as pd

        # Input Spikes
        times = []
        nids = []
        values = []

        num_neurons_str = f"Number of neurons: {self.num_neurons}"
        num_synapses_str = f"Number of synapses: {self.num_synapses}"

        for time in self.input_spikes:
            for nid, value in zip(self.input_spikes[time]["nids"], self.input_spikes[time]["values"]):
                times.append(time)
                nids.append(nid)
                values.append(value)

        input_spikes_df = pd.DataFrame({
            "Time": times,
            "Neuron ID": nids,
            "Value": values
        })

        # Spike train
        spike_train = ""
        for time, spikes in enumerate(self.spike_train):
            spike_train += f"Time: {time}, Spikes: {spikes}\n"

        return (
            num_neurons_str + "\n" + num_synapses_str + "\n"
            "\nNeuron Info: \n"
            + self.neuron_df.to_string(index=False)
            + "\n"
            + "\nSynapse Info: \n"
            + self.synapse_df.to_string(index=False)
            + "\n"
            + "\nSTDP Info: \n"
            + self.stdp_info
            + "\n"
            + "\nInput Spikes: \n"
            + input_spikes_df.to_string(index=False)
            + "\n"
            + "\nSpike Train: \n"
            + spike_train
            + f"\nNumber of spikes: {self.ispikes.sum()}\n"
        )

    def create_neuron(
        self,
        threshold: float = 0.0,
        leak: float = np.inf,
        reset_state: float = 0.0,
        refractory_period: int = 0,
        refractory_state: int = 0,
        initial_state: float = 0.0,
    ) -> Neuron:
        """
        Create a neuron in the neuromorphic model.

        Parameters
        ----------
        threshold : float, optional
            Neuron threshold; the neuron spikes if its internal state is strictly
            greater than the neuron threshold (default is 0.0).
        leak : float, optional
            Neuron leak; the amount by which the internal state of the neuron is
            pushed towards its reset state (default is np.inf).
        reset_state : float, optional
            Reset state of the neuron; the value assigned to the internal state
            of the neuron after spiking (default is 0.0).
        refractory_period : int, optional
            Refractory period of the neuron; the number of time steps for which
            the neuron remains in a dormant state after spiking (default is 0).

        Returns
        -------
        int
            The ID of the created neuron.

        Raises
        ------
        TypeError
            If `threshold`, `leak`, or `reset_state` is not a float or int, or if
            `refractory_period` is not an int.
        ValueError
            If `leak` is less than 0.0 or `refractory_period` is less than 0.

        """
        # Type errors
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be int or float")

        if not isinstance(leak, (int, float)):
            raise TypeError("leak must be int or float")
        if leak < 0.0:
            raise ValueError("leak must be grater than or equal to zero")

        if not is_intlike(refractory_period):
            raise TypeError("refractory_period must be int")
        refractory_period = int(refractory_period)
        if refractory_period < 0:
            raise ValueError("refractory_period must be greater than or equal to zero")

        if not is_intlike(refractory_state):
            raise TypeError("refractory_state must be int")
        refractory_state = int(refractory_state)
        if refractory_state < 0:
            raise ValueError("refractory_state must be greater than or equal to zero")

        if not isinstance(initial_state, (int, float)):
            raise TypeError("initial_state must be int or float")

        # Add neurons to model
        self.neuron_thresholds.append(float(threshold))
        self.neuron_leaks.append(float(leak))
        self.neuron_reset_states.append(float(reset_state))
        self.neuron_refractory_periods.append(refractory_period)
        self.neuron_refractory_periods_state.append(refractory_state)
        self.neuron_states.append(float(initial_state))

        # Return neuron ID
        return Neuron(self, self.num_neurons - 1)

    def create_synapse(
        self,
        pre_id: int | Neuron,
        post_id: int | Neuron,
        weight: float = 1.0,
        delay: int = 1,
        stdp_enabled: bool | Any = False
    ) -> Synapse:
        """Creates a synapse in the neuromorphic model

        Creates synapse connecting a pre-synaptic neuron to a post-synaptic neuron
        with a given set of synaptic parameters (weight, delay and stdp_enabled)

        Parameters
        ----------
        pre_id : int
            ID of the pre-synaptic neuron
        post_id : int
            ID of the post-synaptic neuron
        weight : float
            Synaptic weight; weight is multiplied to the incoming spike (default: 1.0)
        delay : int
            Synaptic delay; number of time steps by which the outgoing signal of the syanpse is delayed by (default: 1)
        enable_stdp : bool
            Boolean value that denotes whether or not STDP learning is enabled on the synapse (default: False)

        Raises
        ------
        TypeError
            if:
                1. pre_id is not an int
                2. post_id is not an int
                3. weight is not a float
                4. delay is not an int
                5. enable_stdp is not a bool

        ValueError
            if:
                1. pre_id is less than 0
                2. post_id is less than 0
                3. delay is less than or equal to 0

        """
        # TODO: deprecate enable_stdp

        # Type errors
        if isinstance(pre_id, Neuron):
            pre_id = pre_id.idx
        if isinstance(post_id, Neuron):
            post_id = post_id.idx

        if not isinstance(pre_id, (int, float)) or not is_intlike(pre_id):
            raise TypeError("pre_id must be int")
        pre_id = int(pre_id)

        if not isinstance(post_id, (int, float)) or not is_intlike(post_id):
            raise TypeError("post_id must be int")
        post_id = int(post_id)

        if not isinstance(weight, (int, float)):
            raise TypeError("weight must be a float")

        if not isinstance(delay, (int, float)) or not is_intlike(delay):
            raise TypeError("delay must be an integer")
        delay = int(delay)

        # Value errors
        if pre_id < 0:
            raise ValueError("pre_id must be greater than or equal to zero")

        if post_id < 0:
            raise ValueError("post_id must be greater than or equal to zero")

        if delay <= 0:
            raise ValueError("delay must be greater than or equal to 1")

        # Collect synapse parameters
        if delay == 1:
            self.pre_synaptic_neuron_ids.append(pre_id)
            self.post_synaptic_neuron_ids.append(post_id)
            self.synaptic_weights.append(weight)
            self.synaptic_delays.append(delay)
            self.enable_stdp.append(stdp_enabled)

        else:
            for _d in range(int(delay) - 1):
                temp_id = self.create_neuron()
                self.create_synapse(pre_id, temp_id)
                pre_id = temp_id

            self.create_synapse(pre_id, post_id, weight=weight, stdp_enabled=stdp_enabled)

        # Return synapse ID
        return Synapse(self, self.num_synapses - 1)

    def add_spike(
        self,
        time: int,
        neuron_id: int | Neuron,
        value: float = 1.0
    ) -> None:
        """Adds an external spike in the neuromorphic model

        Parameters
        ----------
        time : int
            The time step at which the external spike is added
        neuron_id : int
            The neuron for which the external spike is added
        value : float
            The value of the external spike (default: 1.0)

        Raises
        ------
        TypeError
            if:
                1. time is not an int
                2. neuron_id is not an int
                3. value is not an int or float

        """

        # Type errors
        if not is_intlike(time):
            raise TypeError("time must be int")
        time = int(time)

        if isinstance(neuron_id, Neuron):
            neuron_id = neuron_id.idx
        if not is_intlike(neuron_id):
            raise TypeError("neuron_id must be int")
        neuron_id = int(neuron_id)

        if not isinstance(value, (int, float)):
            raise TypeError("value must be int or float")

        # Value errors
        if time < 0:
            raise ValueError("time must be greater than or equal to zero")

        if neuron_id < 0:
            raise ValueError("neuron_id must be greater than or equal to zero")

        # Add spikes
        if time in self.input_spikes:
            self.input_spikes[time]["nids"].append(neuron_id)
            self.input_spikes[time]["values"].append(value)

        else:
            self.input_spikes[time] = {}
            self.input_spikes[time]["nids"] = [neuron_id]
            self.input_spikes[time]["values"] = [value]

    @property
    def apos(self):
        return self._stdp_Apos

    @apos.setter
    def apos(self, value):
        value = np.asarray(value, self.dd)
        if self.stdp_negative_update and len(value) != len(self._stdp_Aneg):  # pyright:ignore[reportArgumentType]
            n = max(len(value), len(self._stdp_Aneg))  # pyright:ignore[reportArgumentType]
            self._stdp_Aneg = resize_vec(self._stdp_Aneg, n)
            value = resize_vec(value, n)
        self._stdp_Apos = value

    @property
    def aneg(self):
        return self._stdp_Aneg

    @aneg.setter
    def aneg(self, value):
        value = np.asarray(value, self.dd)
        if self.stdp_positive_update and len(value) != len(self._stdp_Apos):  # pyright:ignore[reportArgumentType]
            n = max(len(value), len(self._stdp_Apos))  # pyright:ignore[reportArgumentType]
            self._stdp_Apos = resize_vec(self._stdp_Apos, n)
            value = resize_vec(value, n)
        self._stdp_Aneg = value

    def stdp_setup(
        self,
        time_steps: int = 0,
        Apos: list | None = None,
        Aneg: list | None = None,
        positive_update: bool = True,
        negative_update: bool = True,
    ) -> None:
        """Setup the Spike-Time-Dependent Plasticity (STDP) parameters

        Parameters
        ----------
        time_steps : int
            Number of time steps over which STDP learning occurs
        Apos : list, default=None
            List of parameters for excitatory STDP updates
        Aneg : list, default=None
            List of parameters for inhibitory STDP updates
        positive_update : bool
            Boolean parameter indicating whether excitatory STDP update should be enabled
        negative_update : bool
            Boolean parameter indicating whether inhibitory STDP update should be enabled

        Raises
        TypeError
            if:
            1. time_steps is not an int
            2. Apos is not a list
            3. Aneg is not a list
            4. positive_update is not a bool
            5. negative_update is not a bool

        ValueError
            if:
                1. time_steps is less than or equal to zero
                2. Number of elements in Apos is not equal to the time_steps
                3. Number of elements in Aneg is not equal to the time_steps
                4. The elements of Apos are not int or float
                5. The elements of Aneg are not int or float
                6. The elements of Apos are not greater than or equal to 0.0
                7. The elements of Apos are not greater than or equal to 0.0

        RuntimeError
            if:
                1. enable_stdp is not set to True on any of the synapses

        """

        # Type errors
        if not isinstance(time_steps, (int, float)) or not is_intlike(time_steps):
            raise TypeError("time_steps should be int")
        time_steps = int(time_steps)

        # Value error
        if time_steps <= 0:
            raise ValueError("time_steps should be greater than zero")

        if positive_update:
            if not isinstance(Apos, (list, np.ndarray)):
                raise TypeError("Apos should be a list")
            Apos: list
            if len(Apos) != time_steps:
                msg = f"Length of Apos should be {time_steps}"
                raise ValueError(msg)
            if not all([isinstance(x, (int, float)) for x in Apos]):
                raise ValueError("All elements in Apos should be int or float")

        if negative_update:
            if not isinstance(Aneg, (list, np.ndarray)):
                raise TypeError("Aneg should be a list")
            Aneg: list
            if len(Aneg) != time_steps:
                msg = f"Length of Aneg should be {time_steps}"
                raise ValueError(msg)
            if not all([isinstance(x, (int, float)) for x in Aneg]):
                raise ValueError("All elements in Aneg should be int or float")

        # if positive_update and not all([x >= 0.0 for x in Apos]):
        #     raise ValueError("All elements in Apos should be positive")

        # if negative_update and not all([x >= 0.0 for x in Aneg]):
        #     raise ValueError("All elements in Aneg should be positive")

        # Runtime error
        if not any(self.enable_stdp):
            raise RuntimeError("STDP is not enabled on any synapse")

        # Collect STDP parameters
        self.stdp = True
        self._stdp_Apos = Apos
        self._stdp_Aneg = Aneg
        self.stdp_positive_update = positive_update
        self.stdp_negative_update = negative_update

    def setup(self):
        if not self.manual_setup:
            warnings.warn("setup() called without model.manual_setup = True. setup() will be called again in simulate().",
                          RuntimeWarning, stacklevel=2)
        self._setup()

    def weight_mat(self, dtype=None):
        dtype = self.dd if dtype is None else dtype
        if self._use_sparse:
            return csc_array(
                (self.synaptic_weights, (self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids)),
                shape=[self.num_neurons, self.num_neurons]
            )
        mat = np.zeros((self.num_neurons, self.num_neurons), dtype)
        mat[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.synaptic_weights
        return mat

    def stdp_enabled_mat(self, dtype=np.int8):
        if self._use_sparse:
            return csc_array(
                (self.enable_stdp, (self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids)),
                shape=[self.num_neurons, self.num_neurons]
            )
        mat = np.zeros((self.num_neurons, self.num_neurons), dtype)
        mat[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids] = self.enable_stdp
        return mat

    def _setup(self):
        """Setup the neuromorphic model for simulation"""

        # Create numpy arrays for neuron state variables
        self._neuron_thresholds = np.asarray(self.neuron_thresholds, self.dd)
        self._neuron_leaks = np.asarray(self.neuron_leaks, self.dd)
        self._neuron_reset_states = np.asarray(self.neuron_reset_states, self.dd)
        self._neuron_refractory_periods_original = np.asarray(self.neuron_refractory_periods, self.dd)
        self._internal_states = np.asarray(self.neuron_states, self.dd)

        self._neuron_refractory_periods = np.asarray(self.neuron_refractory_periods_state, self.dd)

        # Create numpy arrays for synapse state variables
        self._weights = self.weight_mat()
        anystdp = self.stdp and any(self.enable_stdp)
        self._do_positive_update = anystdp and self.stdp_positive_update and any(self.apos)  # pyright: ignore[reportArgumentType]
        self._do_negative_update = anystdp and self.stdp_negative_update and any(self.aneg)  # pyright: ignore[reportArgumentType]

        self._do_stdp = self._do_positive_update or self._do_negative_update

        # Create numpy arrays for STDP state variables
        if self._do_stdp:
            self._stdp_enabled_synapses = self.stdp_enabled_mat()

            self._stdp_Apos = np.asarray(self.apos, self.dd)
            self._stdp_Aneg = np.asarray(self.aneg, self.dd)

        # Create numpy array for input spikes for each timestep
        self._input_spikes = np.zeros((1, self.num_neurons), self.dd)

        # Create numpy vector for spikes for each timestep
        if len(self.spike_train) > 0:
            self._spikes = np.asarray(self.spike_train[-1], np.int8)
        else:
            self._spikes = np.zeros(self.num_neurons, np.int8)

    def devec(self):
        # De-vectorize from numpy arrays to lists
        self.neuron_states: list[float] = self._internal_states.tolist()
        self.neuron_refractory_periods_state: list[float] = self._neuron_refractory_periods.tolist()

        # Update weights if STDP was enabled
        if self._do_stdp:
            self.synaptic_weights = list(self._weights[self.pre_synaptic_neuron_ids, self.post_synaptic_neuron_ids])

    def zero_neuron_states(self):
        self.neuron_states = np.zeros(self.num_neurons, self.dd).tolist()

    def zero_refractory_periods(self):
        self.neuron_refractory_periods_state = np.zeros(self.num_neurons, self.dd).tolist()

    def clear_spike_train(self):
        self.spike_train = []

    def reset(self):
        # Reset neuromorphic model
        self.zero_neuron_states()
        self.zero_refractory_periods()
        self.clear_spike_train()

    def setup_input_spikes(self, time_steps: int):
        self._input_spikes = np.zeros((time_steps + 1, self.num_neurons), self.dd)
        for t, spikes_dict in self.input_spikes.items():
            if t > time_steps:
                continue
            for neuron_id, amplitude in zip(spikes_dict["nids"], spikes_dict["values"]):
                self._input_spikes[t][neuron_id] = amplitude

    def consume_input_spikes(self, time_steps: int):
        self.input_spikes = {t - time_steps: v for t, v in self.input_spikes.items()
                             if t >= time_steps}

    def release_mem(self):
        """Delete internal variables created during computation. Doesn't delete model."""
        del self._neuron_thresholds, self._neuron_leaks, self._neuron_reset_states, self._internal_states
        del self._neuron_refractory_periods, self._neuron_refractory_periods_original, self._weights
        del self._input_spikes, self._spikes
        if hasattr(self, "_stdp_enabled_synapses"):
            del self._stdp_enabled_synapses
        if hasattr(self, "_output_spikes"):
            del self._output_spikes

    def recommend(self, time_steps: int):
        """Recommend a backend to use based on network size and continuous sim time steps."""
        score = 0
        score += self.num_neurons * 1
        score += time_steps * 100

        if self.gpu and score > self.gpu_threshold:
            return 'gpu'
        elif numba and score > self.jit_threshold:
            return 'jit'
        return 'cpu'

    def simulate(self, time_steps: int = 1, callback=None, use=None, **kwargs) -> None:
        """Simulate the neuromorphic spiking neural network

        Parameters
        ----------
        time_steps : int
            Number of time steps for which the neuromorphic circuit is to be simulated
        callback : function, optional
            Function to be called after each time step, by default None
        use : str, default=None
            Which backend to use. Can be 'cpu', 'jit', or 'gpu'.
            If None, model.backend will be used, which is 'auto' by default.
            'auto' will choose a backend based on the network size and time steps.

        Raises
        ------
        TypeError
            If ``time_steps`` is not an int.
        ValueError
            If ``time_steps`` is less than or equal to zero.
        """

        # Type errors
        if not is_intlike(time_steps):
            raise TypeError("time_steps must be int")
        time_steps = int(time_steps)

        # Value errors
        if time_steps <= 0:
            raise ValueError("time_steps must be greater than zero")

        if use is None:
            use = self.backend

        if isinstance(use, str):
            use = use.lower()
        if use == 'auto':
            use = self.recommend(time_steps)
        if use == 'jit':
            self.simulate_cpu_jit(time_steps, callback, **kwargs)
        elif use == 'gpu':
            self.simulate_gpu(time_steps, callback, **kwargs)
        elif use == 'cpu' or not use:  # if use is falsy, use cpu
            self.simulate_cpu(time_steps, callback, **kwargs)
        else:
            msg = f"Invalid backend: {use}"
            raise ValueError(msg)

    def simulate_cpu_jit(self, time_steps: int = 1, callback=None) -> None:
        # print("Using CPU with Numba JIT optimizations")
        self._last_used_backend = 'jit'
        check_numba()
        if not self.manual_setup:
            self._setup()
            self.setup_input_spikes(time_steps)

        self._spikes = self._spikes.astype(self.dd)

        for tick in range(time_steps):
            if callback is not None:
                if callable(callback):
                    callback(self, tick, time_steps)

            lif_jit(
                tick,
                self._input_spikes,
                self._spikes,
                self._internal_states,  # modified in-place
                self._neuron_thresholds,
                self._neuron_leaks,
                self._neuron_reset_states,
                self._neuron_refractory_periods,
                self._neuron_refractory_periods_original,
                self._weights,
            )

            self.spike_train.append(self._spikes.astype(np.int8))  # COPY
            t = min(self.stdp_time_steps, len(self.spike_train) - 1)
            spikes = np.array(self.spike_train[~t - 1:], dtype=self.dd)

            if self._do_stdp:
                if self._do_positive_update and self._do_negative_update:
                    stdp_update_jit(
                        t,
                        spikes,
                        self._weights,
                        self.apos,
                        self.aneg,
                        self._stdp_enabled_synapses,
                    )
                elif self._do_positive_update:
                    stdp_update_jit_apos(t, spikes, self._weights, self.apos, self._stdp_enabled_synapses)
                elif self._do_negative_update:
                    stdp_update_jit_aneg(t, spikes, self._weights, self.aneg, self._stdp_enabled_synapses)

        if not self.manual_setup:
            self.devec()
            self.consume_input_spikes(time_steps)

    def simulate_cpu(self, time_steps: int = 1000, callback=None) -> None:
        self._last_used_backend = 'cpu'
        if not self.manual_setup:
            self._setup()
            self.setup_input_spikes(time_steps)

        # Simulate
        for tick in range(time_steps):
            if callback is not None:
                if callable(callback):
                    callback(self, tick, time_steps)

            # Leak: internal state > reset state
            indices = self._internal_states > self._neuron_reset_states
            self._internal_states[indices] = np.maximum(
                self._internal_states[indices] - self._neuron_leaks[indices], self._neuron_reset_states[indices]
            )

            # Leak: internal state < reset state
            indices = self._internal_states < self._neuron_reset_states
            self._internal_states[indices] = np.minimum(
                self._internal_states[indices] + self._neuron_leaks[indices], self._neuron_reset_states[indices]
            )

            # # Zero out _input_spikes
            # self._input_spikes -= self._input_spikes

            # # Include input spikes for current tick
            # if tick in self.input_spikes:
            #     self._input_spikes[self.input_spikes[tick]["nids"]] = self.input_spikes[tick]["values"]

            # Internal state
            self._internal_states = self._internal_states + self._input_spikes[tick] + (self._weights.T @ self._spikes)

            # Compute spikes
            self._spikes = np.greater(self._internal_states, self._neuron_thresholds).astype(np.int8)

            # Refractory period: Compute indices of neuron which are in their refractory period
            indices = np.greater(self._neuron_refractory_periods, 0)

            # For neurons in their refractory period, zero out their spikes and decrement refractory period by one
            self._spikes[indices] = 0
            self._neuron_refractory_periods[indices] -= 1

            # For spiking neurons, turn on refractory period
            mask = self._spikes.astype(bool)
            self._neuron_refractory_periods[mask] = self._neuron_refractory_periods_original[mask]

            # Reset internal states
            self._internal_states[mask] = self._neuron_reset_states[mask]

            # Append spike train
            self.spike_train.append(self._spikes)

            # STDP Operations
            t = min(self.stdp_time_steps, len(self.spike_train) - 1)

            if self._do_stdp and t > 0:
                update_synapses = np.outer(
                    np.array(self.spike_train[-t - 1:-1]),
                    np.array(self.spike_train[-1])).reshape([-1, self.num_neurons, self.num_neurons]
                )

                if self._do_positive_update:
                    self._weights += (
                        (update_synapses.T * self._stdp_Apos[0:t][::-1]).T
                    ).sum(axis=0) * self._stdp_enabled_synapses

                if self._do_negative_update:
                    self._weights += (
                        ((1 - update_synapses).T * self._stdp_Aneg[0:t][::-1]).T
                    ).sum(axis=0) * self._stdp_enabled_synapses

        if not self.manual_setup:
            self.devec()
            self.consume_input_spikes(time_steps)

    def simulate_gpu(self, time_steps: int = 1, callback=None) -> None:
        """Simulate the neuromorphic circuit using the GPU backend.

        Parameters
        ----------
        time_steps : int, optional
        callback : _type_, optional
            WARNING: If using the GPU backend, this callback will
            not be able to modify the neuromorphic model via ``self`` .
        """
        # print("Using CUDA GPU via Numba")
        self._last_used_backend = 'gpu'
        check_gpu()
        from .gpu import cuda as gpu
        if self.disable_performance_warnings:
            gpu.disable_numba_performance_warnings()
        # print("Using CPU with Numba JIT optimizations")
        self._output_spikes = np.zeros((time_steps, self.num_neurons), self.dd)
        if not self.manual_setup:
            self._setup()
            self.setup_input_spikes(time_steps)

        if len(self.apos) != len(self.aneg):  # pyright: ignore[reportArgumentType]
            raise ValueError("apos and aneg must be the same length")
        self.stdp_Apos = np.asarray(self.apos, self.dd) if self._do_positive_update else np.zeros(len(self.apos), self.dd)  # pyright: ignore[reportArgumentType]
        self.stdp_Aneg = np.asarray(self.aneg, self.dd) if self._do_negative_update else np.zeros(len(self.aneg), self.dd)  # pyright: ignore[reportArgumentType]

        post_synapse = cuda.to_device(np.zeros(self.num_neurons, self.dd))
        output_spikes = cuda.to_device(self._output_spikes[-1].astype(np.int8))
        states = cuda.to_device(self._internal_states)
        thresholds = cuda.to_device(self._neuron_thresholds)
        leaks = cuda.to_device(self._neuron_leaks)
        reset_states = cuda.to_device(self._neuron_reset_states)
        refractory_periods = cuda.to_device(self._neuron_refractory_periods)
        refractory_periods_original = cuda.to_device(self._neuron_refractory_periods_original)
        weights = cuda.to_device(self._weights)
        if self._do_stdp:
            stdp_enabled = cuda.to_device(self._stdp_enabled_synapses)

        v_tpb = min(self.num_neurons, 32)
        v_blocks = math.ceil(self.num_neurons / v_tpb)

        m_tpb = (v_tpb, v_tpb)
        m_blocks = (v_blocks, v_blocks)

        for tick in range(time_steps):
            if callback is not None:
                if callable(callback):
                    callback(self, tick, time_steps)

            input_spikes = cuda.to_device(self._input_spikes[tick])

            gpu.post_synaptic[v_blocks, v_tpb](
                weights,
                output_spikes,
                post_synapse,
            )

            gpu.lif[v_blocks, v_tpb](
                input_spikes,
                output_spikes,
                post_synapse,
                states,
                thresholds,
                leaks,
                reset_states,
                refractory_periods,
                refractory_periods_original,
            )

            self._output_spikes[tick] = output_spikes.copy_to_host()

            if self._do_stdp:
                for i in range(min(tick, self.stdp_time_steps)):
                    old_spikes = cuda.to_device(self._output_spikes[tick - i - 1].astype(np.int8))
                    gpu.stdp_update[m_blocks, m_tpb](weights, old_spikes, output_spikes,
                                    stdp_enabled, self._stdp_Apos[i], self._stdp_Aneg[i])

        self.spike_train.extend(self._output_spikes)
        self._weights = weights.copy_to_host()
        self._neuron_refractory_periods = refractory_periods.copy_to_host()
        self._internal_states = states.copy_to_host()

        self.devec()

    def print_spike_train(self):
        """Prints the spike train."""

        for time, spike_train in enumerate(self.spike_train):
            print(f"Time: {time}, Spikes: {spike_train}")

        # print(f"\nNumber of spikes: {self.num_spikes}\n")

    def copy(self):
        """Returns a copy of the neuromorphic model"""

        return copy.deepcopy(self)
