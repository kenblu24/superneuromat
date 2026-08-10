from dataclasses import dataclass

from ..neuromorphicmodel import SNN
from ..accessor_classes import Neuron, Synapse, mlist, asmlist
from ..accessor_classes import NeuronListView, SynapseListView


class CaspianImporter:
    def __init__(self, filename):
        self.filename = filename
        self.snn = SNN()
        self.inputs = NeuronListView(self.snn, [])
        self.outputs = NeuronListView(self.snn, [])
        self.hidden = NeuronListView(self.snn, [])
        self.synapses = SynapseListView(self.snn, [])
        self.j = {}
        self.keep_neuron_ids = True
        self.max_neurons = None
        self.node_propmap = {}
        self.edge_propmap = {}
        self.node_id_map = {}  # map node IDs to SNN neuron IDs

    def set_neuron_props(self, nodes):
        snn = self.snn
        nmap = self.node_propmap
        for idx, props in nodes:
            snn.neurons[idx].threshold = props[nmap["Threshold"]]
            snn.neurons[idx].leak = props[nmap["Leak"]]
            self.node_id_map[idx] = idx

    def make_neuron_with_props(self, props):
        snn = self.snn
        nmap = self.node_propmap
        neuron = snn.create_neuron(
            threshold=props[nmap["Threshold"]],
            leak=props[nmap["Leak"]],
        )
        return neuron

    def make_synapse_with_props(self, edge):
        snn = self.snn
        emap = self.edge_propmap
        props = edge['values']
        return snn.create_synapse(
            self.node_id_map[edge['from']],
            self.node_id_map[edge['to']],
            weight=props[emap["Weight"]],
            delay=props[emap["Delay"]],
        )

    @staticmethod
    def mapping(props: list[dict]):
        return {prop['name']: prop['index'] for prop in props}

    def network_from_json(self, j: dict) -> tuple[dict[int, Node], list[Node], list[Node]]:
        # read a Tennlab json network and create it.

        # get mapping of property name to index in 'values' list i.e. m_n['Delay'] -> 1
        # need this because the network json represents the node/edge params as an
        # unordered list i.e. 'values': [127, -1, 0] <-- threshold, leak, delay
        self.node_propmap = self.mapping(j['Properties']['node_properties'])
        self.edge_propmap = self.mapping(j['Properties']['edge_properties'])

        # make nodes from json
        j_nodes = sorted(j['Nodes'], key=lambda v: v['id'])
        nodes = [(n['id'], n['values']) for n in j_nodes]
        if self.keep_neuron_ids:
            highest_id, _params = max(nodes, key=lambda v: v[0])
            for _i in range(highest_id + 1):
                self.snn.create_neuron()
            self.set_neuron_props(nodes)
        else:
            for idx, props in nodes:
                neuron = self.make_neuron_with_props(props)
                self.node_id_map[idx] = neuron.idx

        # make connections from json
        self.synapses = [self.make_synapse_with_props(e) for e in j['Edges']]

        return self.snn


class CaspianExporter:
    def __init__(self, snn: SNN):
        self.snn = snn
        self.j = {}
        self.keep_neuron_ids = True
        self.max_neurons = None
        self.node_propmap = {}
        self.edge_propmap = {}
        self.node_id_map = {}  # map node IDs to SNN neuron IDs
        # self.remove_
