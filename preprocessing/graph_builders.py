import abc
import collections
from functools import lru_cache
from time import time
from copy import deepcopy
from platform import node
import typing as tp

import numpy as np
from py import process
import torch
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.objects.log import obj
from pm4py.util import xes_constants as xes_util
from torch_geometric import data as tg_data_utils
from torch_geometric.utils import from_networkx

from preprocessing import preprocessors
from preprocessing import process_graph as pg
import constants
from constants import *

import networkx


activity_key = xes_util.DEFAULT_NAME_KEY
timestamp_key = xes_util.DEFAULT_TIMESTAMP_KEY


ALL_EDGES = 'all'
ONLY_OLD_EDGES = 'only_old'
ONLY_NEW_EDGES = 'only_new'

class StaticGraphBuilder(preprocessors.BasePreprocessor):
    def __init__(
        self,
        activity2id: tp.Dict[str, int],
        event_log: obj.EventLog,
        graph_type: str,
        sequential: bool,
        add_edges_type: str,
        key=None,
    ):
        if sequential:
            key='sequential_graph'
        else:
            key='process_graph',
        super().__init__(key)
        if graph_type == 'dfg':
            self.process_graph = pg.ProcessGraph.build_from_evlog_dfg(
                event_log,
                activity2id,
            )
        elif graph_type == 'bpmn':
            self.process_graph = pg.ProcessGraph.build_from_evlog_bpmn(
                event_log,
                activity2id,
            )
        else:
            raise NotImplementedError
        self.sequential = sequential
        self.add_edges = add_edges_type
        assert self.add_edges in [ALL_EDGES, ONLY_NEW_EDGES, ONLY_OLD_EDGES]

    def make_tg_data(self, graph):
        return from_networkx(graph, group_node_attrs=['node_type'], group_edge_attrs=['edge_type', 'frequency', 'time'])
    
    def get_trace_changed_graph(
        self,
        trace: obj.Trace,
        times: tp.List[float],
        activity_ids: tp.List[int],
    ):
        graph = self.process_graph.graph.copy()
        path = self.process_graph.get_path_in_graph_corresp_to_trace(trace)
        if self.add_edges == ONLY_NEW_EDGES:
            graph.clear_edges()
        j = 0
        if self.add_edges != ONLY_OLD_EDGES:
            edges = [pg.create_edge(0, 0, pg.DISCOVERED, 1, 0.0)]
            for i in range(len(path) - 1):
                fr = path[i]    
                to = path[i + 1]
                if to == activity_ids[j + 1]:
                    j += 1
                e = pg.create_edge(fr, to, pg.APPEARED, 1, times[j])
                graph.add_edge(
                    fr,
                    to,
                    edge_type=e[2]['edge_type'],
                    frequency=e[2]['frequency'],
                    time=e[2]['time']
                )
                edges.append(pg.create_edge(
                    fr, to,
                    pg.APPEARED,
                    1,
                    times[j]
                ))
            graph.add_edges_from(edges)
            assert activity_ids[j] == path[-1]
            assert j == len(activity_ids) - 1

        return graph

    def process_trace(
        self,
        trace: obj.Trace,
        times: tp.List[float],
        activity_ids: tp.List[int],
    ):
        changed_graph = self.get_trace_changed_graph(trace, times, activity_ids) 
        return self.make_tg_data(changed_graph)
    
    def visualize_trace(
        self,
        trace: obj.Trace,
        times: tp.List[float],
        activity_ids: tp.List[float],
    ):
        graph = self.get_trace_changed_graph(trace, times, activity_ids) 
        subg = graph.subgraph([node for node in graph][1:])
        agraph = networkx.nx_agraph.to_agraph(subg)
        agraph.node_attr['shape'] = 'box'
        agraph.get_node(1).attr['color'] = 'green'
        agraph.get_node(2).attr['color'] = 'red'
        agraph.get_node(1).attr['shape'] = 'circle'
        agraph.get_node(2).attr['shape'] = 'circle'
        for node in subg.nodes:
            agraph.get_node(node).attr['label'] = subg.nodes[node]['node_type_name']
        for edge in subg.edges:
            edge_d = subg.get_edge_data(edge[0], edge[1], edge[2])
            cl = 'black' if edge_d['edge_type'] else 'gray'
            label = f"{edge_d['time']:.4f}"
            agraph.get_edge(edge[0], edge[1], edge[2]).attr['color'] = cl
            if cl == 'black':
                agraph.get_edge(edge[0], edge[1], edge[2]).attr['label'] = label
            agraph.get_edge(edge[0], edge[1], edge[2]).attr['fontsize'] = 4
            
        agraph.layout(prog='dot')
        agraph.draw('bpmn_graph.pdf')


    def __call__(self, data_item: tp.Dict[str, tp.Any]) -> tp.Any:
        trace = data_item['trace']
        times = data_item[TIME_SEQUENCE]
        activity_ids = data_item[ACTIVITIES_SEQUENCE]
        if self.sequential:
            res = [
                self.process_trace(trace[:i], times[:i + 1], activity_ids[:i + 1])
                for i in range(len(trace))
            ]
            return res
        else:
            return self.process_trace(trace, times, activity_ids)       

