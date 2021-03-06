import numpy as np
from typing import *
import scipy.sparse as sp
from collections.abc import Iterable


class Graph:
    """
    A container to represent a graph. The data associated with the Graph is
    stored in its attributes:

        - `x`, for the node features;
        - `a`, for the adjacency matrix;
        - `e`, for the edge attributes;
        - `y`, for the node or graph labels;

    All of these default to `None` if you don't specify them in the constructor.
    If you want to read all non-None attributes at once, you can call the
    `numpy()` method, which will return all data in a tuple (with the order
    defined above).

    Graphs also have the following attributes that are computed automatically
    from the data:

    - `n_nodes`: number of nodes;
    - `n_edges`: number of edges;
    - `n_node_features`: size of the node features, if available;
    - `n_edge_features`: size of the edge features, if available;
    - `n_labels`: size of the labels, if available;

    Any additional `kwargs` passed to the constructor will be automatically
    assigned as instance attributes of the graph.

    Data can be stored in Numpy arrays or Scipy sparse matrices, and labels can
    also be scalars.

    Spektral usually assumes that the different data matrices have specific
    shapes, although this is not strictly enforced to allow more flexibility.
    In general, node attributes should have shape `(n_nodes, n_node_features)` and the adjacency
    matrix should have shape `(n_nodes, n_nodes)`.

    Edge attributes can be stored in a dense format as arrays of shape
    `(n_nodes, n_nodes, n_edge_features)` or in a sparse format as arrays of shape `(n_edges, n_edge_features)`
    (so that you don't have to store all the zeros for missing edges). Most
    components of Spektral will know how to deal with both situations
    automatically.

    Labels can refer to the entire graph (shape `(n_labels, )`) or to each
    individual node (shape `(n_nodes, n_labels)`).

    **Arguments**

    - `x`: np.array, the node features (shape `(n_nodes, n_node_features)`);
    - `a`: np.array or scipy.sparse matrix, the adjacency matrix (shape `(n_nodes, n_nodes)`);
    - `e`: np.array, the edge features (shape `(n_nodes, n_nodes, n_edge_features)` or `(n_edges, n_edge_features)`);
    - `y`: np.array, the node or graph labels (shape `(n_nodes, n_labels)` or `(n_labels, )`);


    """

    def __init__(self, x=None, a=None, e=None, y=None, **kwargs):
        self.x = x
        self.a = a
        self.e = e
        self.y = y

        # Read extra kwargs
        for k, v in kwargs.items():
            self[k] = v

    def numpy(self):
        return tuple(ret for ret in [self.x, self.a, self.e, self.y] if ret is not None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __contains__(self, key):
        return key in self.keys

    def __repr__(self):
        return "Graph(n_nodes={}, n_node_features={}, n_edge_features={}, n_labels={})".format(
            self.n_nodes, self.n_node_features, self.n_edge_features, self.n_labels
        )

    @property
    def n_nodes(self):
        if self.x is not None:
            return self.x.shape[-2]
        elif self.a is not None:
            return self.a.shape[-1]
        else:
            return None

    @property
    def n_edges(self):
        if sp.issparse(self.a):
            return self.a.nnz
        elif isinstance(self.a, np.ndarray):
            return np.count_nonzero(self.a)
        else:
            return None

    @property
    def n_node_features(self):
        if self.x is not None:
            return self.x.shape[-1]
        else:
            return None

    @property
    def n_edge_features(self):
        if self.e is not None:
            return self.e.shape[-1]
        else:
            return None

    @property
    def n_labels(self):
        if self.y is not None:
            shp = np.shape(self.y)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None

    @property
    def keys(self):
        keys = [
            key
            for key in self.__dict__.keys()
            if self[key] is not None and not key.startswith("__")
        ]
        return keys

class GraphSnapshot(Graph):
    def __init__(self, x=None, a=None, e=None, y=None, t=None, **kwargs):
        self.t = t
        super().__init__(
        x = x,
        a = a,
        e = e,
        y = y,
        **kwargs)


class DynamicGraph: 
    """
    A container to represent a dynamic graph, a sequence of graphs with each assigned a timestamp. The data associated
    with the DynamicGraph is stored in its attributes:
   
   params:
       nodes:  N * H or N * H * T ndarray, where T is number of timetamps, N is number of nodes, H is number of
       features or node labels.  If T is not present, node features is fixed for all timestamp 
       edges: a dict with key to be the timestamp and the value is a list of edges, where edge[0] and edge[1] are the
       node id that could be found in the attribute nodes, and edge[2:] to be the edge features
    """
    def __init__(self, nodes:np.ndarray, edges: Dict[float, List[float]], node_id_maps=None):
        self.nodes = nodes
        self.edges = edges

    def __sort_key(snapshot: GraphSnapshot) -> int:
        return int(snapshot.t)
    
    def n_timestamp(self) -> int:
        return len(self.edges)

    def irregular_node_feature(self) -> bool:
        return (len(self.nodes.shape) == 3)

    def timestamps(self) -> List:
        return sorted(self.edges.keys(), reverse=False)

    def n_edge_features(self) -> int:
        return len(self.edges[0][0])-2

    @property
    def n_nodes(self) -> int:
        """
        return the maximum size of the snapshots
        """
        return self.nodes.shape[0]

    @property
    def n_node_features(self) -> int:
        return self.nodes.shape[1]
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __contains__(self, key):
        return key in self.keys
    def __contains__(self, key):
        return key in self.keys
    
    def __repr__(self):
        return "Graph(n_nodes={}, n_node_features={}, n_edge_features={}, n_labels={})".format(
            self.n_nodes, self.n_node_features, self.n_edge_features, self.n_labels
        )
    @property
    def keys(self):
        keys = [
            key
            for key in self.__dict__.keys()
            if self[key] is not None and not key.startswith("__")
        ]
        return keys
    
    # generator to generate snapshots of the dynamic graph
    # todo add support for directed graph
    def get_snapshots(self, n_snapshot: int, positional = True, transform=None):
        time_map = {}
        timestamps_list = self.timestamps()
        n_timestamp = self.n_timestamp()
        print(n_timestamp)
        if positional: 
            timestamp_per_shot = int(n_timestamp / n_snapshot)
            print(timestamp_per_shot)
            for i in range(n_snapshot):
                start = i*timestamp_per_shot
                end = (i+1)*timestamp_per_shot
                if end >= n_timestamp:
                    time_map[i] = timestamps_list[start:]
                else:
                    time_map[i] = timestamps_list[start: end]
        else:
            duration = (max(timestamps_list) - min(timestamps_list))/ n_snapshot
            print(duration)
            for i in range(n_snapshot):
                time_map[i] = [t for t in timestamps_list if t >= (i*duration) and t < ((i+1)*duration)]

        cummulated_e = None

        nodes_feature = None
        if not self.irregular_node_feature():
            nodes_feature = self.nodes
        else:
            nodes_feature = np.zeros(self.nodes.shape[:2])
        if self.n_edge_features() > 1:
            cummulated_e = np.zeros([self.n_nodes, self.n_nodes, self.n_edge_features])
        elif self.n_edge_features() == 1:
            cummulated_e = np.zeros([self.n_nodes, self.n_nodes])
        else:
            cummulated_e = None

        cummulated_a = np.zeros([self.n_nodes, self.n_nodes])
        for i in range(n_snapshot):
            for tm in time_map[i]:
                if self.irregular_node_feature():
                    pass # todo add support for dynamic graph with node features changing along time
                for e in self.edges[tm]:
                    cummulated_a[e[0], e[1]] = 1
                    cummulated_a[e[1], e[0]] = 1
                    if cummulated_e is not None:
                        if self.n_edge_features() > 1:
                            for j in range(self.n_edge_features()):
                                cummulated_e[e[0], e[1], j] = e[2+j]
                                cummulated_e[e[1], e[0], j] = e[2+j]
                        else:
                            cummulated_e[e[0], e[1]] = e[2]
                            cummulated_e[e[1], e[0]] = e[2]

            current_g = GraphSnapshot(x=self.nodes,e = cummulated_e, a=cummulated_a, t=i)
            if transform:
                if not callable(transform):
                    raise ValueError("`transform` must be callable")
                current_g = transform(current_g)
            yield current_g


