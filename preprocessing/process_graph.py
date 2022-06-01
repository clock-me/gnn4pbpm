from copy import deepcopy
from tkinter import W
import typing as tp
from pm4py.objects.log import obj
import pm4py
import constants
import enum
import networkx


DISCOVERED = 0
APPEARED = 1


def create_edge(
    fr: int,
    to: int,
    edge_type: int,
    frequency: int,
    time: float = 0.0):
    return (
        fr,
        to,
        {
            'edge_type': edge_type,
            'frequency': frequency,
            'time': time,
        },
    )

class ProcessGraph:
    def __init__(
        self,
        activity_to_id: tp.Dict[str, int],
        graph: networkx.MultiDiGraph,
    ) -> None:
        self.activity2id = activity_to_id
        self.graph = graph
        self.shortest_path_cache = {}

    @classmethod
    def build_from_evlog_dfg(cls, event_log: obj.EventLog,  activity2id: tp.Dict[str, int]):
        dfg, start_activities, end_activities = pm4py.discover_dfg(event_log) 
        edges = []
        for start_activity in start_activities:
            edges.append(
                create_edge(
                    fr=constants.BOS_i,
                    to=activity2id[start_activity],
                    edge_type=DISCOVERED,
                    frequency=1,
                    time=0.0
                )
            )
        for fr_a, to_a in dfg:
            edges.append(create_edge(
                fr=activity2id[fr_a],
                to=activity2id[to_a],
                edge_type=DISCOVERED,
                frequency=dfg[(fr_a, to_a)],
                time=0.0,
            )) 
        
        for end_activity in end_activities:
            edges.append(
                create_edge(
                    fr=activity2id[end_activity],
                    to=constants.EOS_i,
                    edge_type=DISCOVERED,
                    frequency=end_activities[end_activity],
                    time=0.0
                )
            )
        result = [(i, {'node_type_name': None, 'node_type': None, 'name': None, 'i': i}) for i in range(len(activity2id))]
        for activity in activity2id:
            result[activity2id[activity]][1]['node_type_name'] = activity
            result[activity2id[activity]][1]['node_type'] = activity2id[activity]
            result[activity2id[activity]][1]['name'] = activity
        graph = networkx.MultiDiGraph()
        graph.add_nodes_from(result)
        graph.add_edges_from(edges)
        return cls(activity2id, graph)

    @classmethod
    def build_from_evlog_bpmn(cls, event_log: obj.EventLog, activity2id: tp.Dict[str, int]):
        bpmn = pm4py.discover_bpmn_inductive(event_log)

        name2node = {}

        def get_node_type_name(name):
            if name == 'start':
                type_name = constants.BOS
            elif name == 'end':
                type_name = constants.EOS
            elif name in activity2id:
                type_name = name
            else:
                splitted = name.split('_')
                assert len(splitted) == 3
                assert splitted[1].isdigit()
                type_name = splitted[0] + '_' + splitted[2]
            return type_name

        max_node = len(activity2id)
        type2id = deepcopy(activity2id)
        max_type = len(type2id)
        name2type = {}
        name2type_name = {}
        for node in bpmn.get_nodes():
            name = node.get_name()
            if name in activity2id:
                name2node[name] = activity2id[name]
            elif name == 'start':
                name2node[name] = constants.BOS_i
            elif name == 'end':
                name2node[name] = constants.EOS_i
            else:
                name2node[name] = max_node
                max_node += 1
            type_name = get_node_type_name(name)
            name2type_name[name] = type_name
            if type_name in type2id:
                name2type[name] = type2id[type_name] 
            else:
                name2type[name] = max_type
                type2id[type_name] = max_type
                max_type += 1
        
        nodes = [(i, {'name': None, 'node_type': None, 'node_type_name': None, 'i': i}) for i in range(max_node)]
        nodes[0][1]['name'] = constants.PAD
        nodes[0][1]['node_type'] = constants.PAD_i
        nodes[0][1]['node_type_name'] = constants.PAD
        for node_name in name2node:
            nodes[name2node[node_name]][1]['name'] = node_name
            nodes[name2node[node_name]][1]['node_type'] = name2type[node_name]
            nodes[name2node[node_name]][1]['node_type_name'] = name2type_name[node_name]

        edges = [
            create_edge(
                name2node[edge[0].get_name()],
                name2node[edge[1].get_name()],
                DISCOVERED,
                1,
                0.0
            )
            for edge in bpmn.get_graph().edges
        ]
        graph = networkx.MultiDiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return cls(activity2id, graph)

    def get_path_in_graph_corresp_to_trace(self, trace: obj.Trace):
        vertices_i_in_path = [1]
        for event in trace:
            activity_slug = event['concept:name']
            current_vertice = vertices_i_in_path[-1]
            dest_vertice = self.activity2id[activity_slug]
            if self.graph.has_edge(current_vertice, dest_vertice):
                vertices_i_in_path.append(dest_vertice)
                continue
            if networkx.has_path(self.graph, current_vertice, dest_vertice):
                path = networkx.shortest_path(self.graph, current_vertice, dest_vertice) 
                assert path[0] == current_vertice 
                assert path[-1] == dest_vertice
                activity_meet = False
                for j in path[1:-1]:
                    if j in self.activity2id:
                        activity_meet = True
                        break
                if activity_meet:
                    vertices_i_in_path.append(dest_vertice)
                else:
                    vertices_i_in_path.extend(path if len(path) == 1 else path[1:])
            else:
                vertices_i_in_path.append(dest_vertice)
        return vertices_i_in_path