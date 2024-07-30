from itertools import permutations, combinations
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

class MixedDiGraph:
    def __init__(self, hyper=False, multi=False, adirected=False, directed=False):
        self.nodes = []
        self.plotted_coordinates = {}
        if hyper or multi:
            raise NotImplementedError('Hypergraphs and multigraphs are not supported yet.')
        self.hyper = hyper
        self.multi = multi
        self.adirected = adirected
        self.directed = directed
        self.incidence_matrix = np.zeros((0, 0))

    def add_node(self, node):
        if self.directed:
            raise ValueError("Cannot add isolated nodes in a directed graph.")
        if node not in self.nodes:
            self.nodes.append(node)
            # Expand the incidence matrix to accommodate the new node
            new_row = np.zeros((1, self.incidence_matrix.shape[1]))
            self.incidence_matrix = np.vstack([self.incidence_matrix, new_row])

    def add_edge(self, nodes):
        if not isinstance(nodes, (list, tuple)):
            raise ValueError("Nodes must be provided as a list or tuple of nodes.")
        if not self.hyper and len(nodes) > 2:
            raise ValueError("Cannot add hyper edges in a non-hypergraph.")
        if self.adirected and isinstance(nodes, tuple):
            raise ValueError("Cannot add directed edges in an adirected graph.")
        if self.directed and not isinstance(nodes, tuple):
            raise ValueError("Cannot add undirected edges in a directed graph.")

        if isinstance(nodes, tuple):
            u, v = nodes
            if not self.multi and (u, v) in self.get_directed_edges() or (v, u) in self.get_undirected_edges():
                raise ValueError(f"Multiple edges between {u} and {v} are not allowed in a non-multigraph.")
            for node in nodes:
                if node not in self.nodes:
                    self.nodes.append(node)
                    new_row = np.zeros((1, self.incidence_matrix.shape[1]))
                    self.incidence_matrix = np.vstack([self.incidence_matrix, new_row])
        else:  # Unordered list of nodes
            for node in nodes:
                if node not in self.nodes:
                    self.nodes.append(node)
                    new_row = np.zeros((1, self.incidence_matrix.shape[1]))
                    self.incidence_matrix = np.vstack([self.incidence_matrix, new_row])

        # Expand the incidence matrix to accommodate the new edge
        new_col = np.zeros((self.incidence_matrix.shape[0], 1))
        self.incidence_matrix = np.hstack([self.incidence_matrix, new_col])

        edge_index = self.incidence_matrix.shape[1] - 1

        if isinstance(nodes, tuple):
            u, v = nodes
            u_index = self.nodes.index(u)
            v_index = self.nodes.index(v)
            self.incidence_matrix[u_index, edge_index] = 0.5
            self.incidence_matrix[v_index, edge_index] = -0.5
        else:  # Unordered list of nodes
            for node in nodes:
                node_index = self.nodes.index(node)
                self.incidence_matrix[node_index, edge_index] = 1

    def remove_node(self, node):
        if node not in self.nodes:
            raise ValueError(f"Node {node} does not exist in the graph.")
        
        index = self.nodes.index(node)
        self.nodes.remove(node)
        self.incidence_matrix = np.delete(self.incidence_matrix, index, axis=0)
        
        # Remove edges connected to this node
        cols_to_delete = []
        for col in range(self.incidence_matrix.shape[1]):
            if np.any(self.incidence_matrix[:, col] != 0):
                cols_to_delete.append(col)
        
        self.incidence_matrix = np.delete(self.incidence_matrix, cols_to_delete, axis=1)

    def remove_edge(self, nodes):
        if isinstance(nodes, tuple):
            u, v = nodes
            if (u, v) not in self.get_directed_edges():
                raise ValueError(f"Edge between {u} and {v} does not exist.")
            u_index = self.nodes.index(u)
            v_index = self.nodes.index(v)
            col_index = np.where((self.incidence_matrix[u_index] == 0.5) & (self.incidence_matrix[v_index] == -0.5))[0]
        else:
            if not self.hyper and len(nodes) != 2:
                raise ValueError("Undirected edge must connect exactly two nodes.")
            node_indices = [self.nodes.index(node) for node in nodes]
            col_index = np.where(np.all(self.incidence_matrix[node_indices] == 1, axis=0))[0]
        
        if len(col_index) == 0:
            raise ValueError(f"Edge {nodes} does not exist.")
        
        self.incidence_matrix = np.delete(self.incidence_matrix, col_index[0], axis=1)

    def _to_networkx(self):
        G = nx.Graph()
        node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        
        for node in self.nodes:
            G.add_node(node)
        
        # run get_directed_edges() and get_undirected_edges() to get the edges
        for edge in self.get_directed_edges():
            print("Directed edge:", edge)
            G.add_edge(edge[0], edge[1])
        for edge in self.get_undirected_edges():
            print("Undirected edge:", edge)
            G.add_edge(edge[0], edge[1])
        return G

    def get_undirected_edges(self, nodes=None):
        undirected_edges = []
        for edge_index in range(self.incidence_matrix.shape[1]):
            node_indices = np.where(self.incidence_matrix[:, edge_index] == 1)[0]
            if len(node_indices) == 2:
                u, v = self.nodes[node_indices[0]], self.nodes[node_indices[1]]
                if nodes is None or (u in nodes and v in nodes):
                    undirected_edges.append((u, v))
        return undirected_edges

    def get_directed_edges(self, nodes=None):
        directed_edges = []
        for edge_index in range(self.incidence_matrix.shape[1]):
            from_node = np.where(self.incidence_matrix[:, edge_index] == 0.5)[0]
            to_node = np.where(self.incidence_matrix[:, edge_index] == -0.5)[0]
            if len(from_node) == 1 and len(to_node) == 1:
                u, v = self.nodes[from_node[0]], self.nodes[to_node[0]]
                if nodes is None or (u in nodes and v in nodes):
                    directed_edges.append((u, v))
        return directed_edges

    def plot(self):
        G = nx.Graph()

        # Add nodes
        for node in self.nodes:
            G.add_node(node)

        # Add edges
        directed_edges = self.get_directed_edges()
        undirected_edges = self.get_undirected_edges()

        for u, v in undirected_edges:
            G.add_edge(u, v)

        self.plotted_coordinates = nx.kamada_kawai_layout(G)

        plt.figure(figsize=(12, 8))
        nx.draw(G, self.plotted_coordinates, with_labels=True, node_size=700, node_color='skyblue', font_size=16, font_color='darkblue')

        # Add arrows for directed edges
        ax = plt.gca()
        for u, v in directed_edges:
            ax.annotate("",
                        xy=self.plotted_coordinates[v], xycoords='data',
                        xytext=self.plotted_coordinates[u], textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        color="black",
                                        shrinkA=15, shrinkB=15,  # Adjust these values to avoid overlap
                                        patchA=None,
                                        patchB=None,
                                        connectionstyle="arc3,rad=0.1",
                                        ),
                        )
        plt.show()

    def export(self, output_folder='output_images', filename='graph_plot.png'):
        if not self.plotted_coordinates:
            print("Graph not yet plotted, plotting now...")
            self.plot()

        G = nx.Graph()

        # Add nodes
        for node in self.nodes:
            G.add_node(node)

        # Add edges
        directed_edges = self.get_directed_edges()
        undirected_edges = self.get_undirected_edges()

        for u, v in undirected_edges:
            G.add_edge(u, v)

        plt.figure(figsize=(12, 8))
        nx.draw(G, self.plotted_coordinates, with_labels=True, node_size=700, node_color='skyblue', font_size=16, font_color='darkblue')

        # Add arrows for directed edges
        ax = plt.gca()
        for u, v in directed_edges:
            ax.annotate("",
                        xy=self.plotted_coordinates[v], xycoords='data',
                        xytext=self.plotted_coordinates[u], textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        color="black",
                                        shrinkA=15, shrinkB=15,  # Adjust these values to avoid overlap
                                        patchA=None,
                                        patchB=None,
                                        connectionstyle="arc3,rad=0.1",
                                        ),
                        )

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the plot to the specified file in the output folder
        plt.savefig(os.path.join(output_folder, filename))
        plt.close()

    def is_isomorphic(self, other):
        if len(self.nodes) != len(other.nodes) or self.incidence_matrix.shape != other.incidence_matrix.shape:
            return False

        # Generate all permutations of node mappings
        node_permutations = permutations(other.nodes)
        for perm in node_permutations:
            mapping = {self_node: other_node for self_node, other_node in zip(self.nodes, perm)}

            # Check if the adjacency matrix is preserved under the mapping
            if self._check_isomorphism(mapping, other):
                return True
        return False

    def _check_isomorphism(self, mapping, other):
        for i, node in enumerate(self.nodes):
            mapped_index = other.nodes.index(mapping[node])
            if not np.array_equal(self.incidence_matrix[i], other.incidence_matrix[mapped_index]):
                return False
        return True

    def _has_isomorphic_subgraph(self, subgraph):
        nx_graph = self._to_networkx()
        GM = nx.algorithms.isomorphism.GraphMatcher(nx_graph, subgraph)
        return GM.subgraph_is_isomorphic()

    def find_isomorphic_subgraphs(self, subgraph):
        subgraph_nodes_count = len(subgraph.nodes)
        subgraph_inc_matrix = subgraph.incidence_matrix

        def is_subgraph_isomorphic(subgraph_matrix, candidate_matrix):
            for perm in permutations(range(candidate_matrix.shape[0]), subgraph_matrix.shape[0]):
                permuted_matrix = candidate_matrix[np.ix_(perm, perm)]
                if np.array_equal(subgraph_matrix, permuted_matrix):
                    return True
            return False

        isomorphic_subgraphs = []
        for nodes_combination in combinations(range(len(self.nodes)), subgraph_nodes_count):
            candidate_matrix = self.incidence_matrix[np.ix_(nodes_combination, nodes_combination)]
            if is_subgraph_isomorphic(subgraph_inc_matrix, candidate_matrix):
                isomorphic_subgraphs.append([self.nodes[i] for i in nodes_combination])
        
        return isomorphic_subgraphs

# Example usage
if __name__ == "__main__":
    graph1 = MixedDiGraph()
    graph1.add_edge(('A', 'B'))  # Directed edge from A to B
    graph1.add_edge(('B', 'C'))  # Directed edge from B to C
    graph1.add_edge(('C', 'A'))  # Directed edge from C to A
    graph1.add_edge(('A', 'D'))  # Directed edge from A to D

    subgraph = MixedDiGraph()
    subgraph.add_edge(('X', 'Y'))  # Directed edge from X to Y
    subgraph.add_edge(('Y', 'Z'))  # Directed edge from Y to Z
    subgraph.add_edge(('Z', 'X'))  # Directed edge from Z to X

    print("Are the two graphs isomorphic?", graph1.is_isomorphic(subgraph))

    print("Isomorphic subgraphs of the subgraph in graph1:", graph1.find_isomorphic_subgraphs(subgraph))
