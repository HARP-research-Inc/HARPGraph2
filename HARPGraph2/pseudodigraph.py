import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

class PseudoDiGraph:
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
            if not self.multi and (u, v) in self.get_directed_edges():
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

"""# Example usage
graph = PseudoDiGraph(directed=True)
graph.add_edge(('A', 'B'))  # Directed edge from A to B
print("Directed edges:", graph.get_directed_edges())
graph.plot()  # Displays the plot
graph.export()  # Saves the plot to the default 'output_images/graph_plot.png'

graph2 = PseudoDiGraph()
graph2.add_node('C')
graph2.add_node('D')
graph2.add_edge(('C', 'D'))  # Unordered hyper edge between C and D
graph2.add_node('E')
graph2.add_node('F')
graph2.add_edge(('D', 'E'))  # Directed edge from D to E
graph2.add_edge(('E', 'C'))  # Directed edge from E to D
graph2.add_edge(['C', 'F'])  # Directed edge from C to F
print("Undirected edges:", graph2.get_undirected_edges())
print("Directed edges:", graph2.get_directed_edges())
graph2.plot()  # Displays the plot
graph2.export(output_folder="output_images")  # Saves the plot to the default 'output_images/graph_plot.png'

# Removing nodes and edges
graph2.remove_edge(('C', 'D'))
graph2.remove_node('E')
print("Undirected edges after removal:", graph2.get_undirected_edges())
print("Directed edges after removal:", graph2.get_directed_edges())
graph2.plot()  # Displays the plot after removals
graph2.export(output_folder="output_images", filename="graph_plot_after_removal.png")  # Saves the plot to the default 'output_images/graph_plot_after_removal.png'
"""