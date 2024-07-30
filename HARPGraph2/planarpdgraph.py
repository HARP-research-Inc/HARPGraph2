import networkx as nx
import numpy as np
import sys, os
from matplotlib import pyplot as plt

# Check if pytest is running
is_pytest_running = "pytest" in sys.modules

if is_pytest_running:
    from HARPGraph2.pseudodigraph import PseudoDiGraph
else:
    from pseudodigraph import PseudoDiGraph

class PlanarPDGraph(PseudoDiGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.faces = {}  # List to store faces as dicts
    
    def add_edge(self, edge):
        super().add_edge(edge)
    
    def is_planar(self):
        return self._is_planar()

    def _is_planar(self):
        if self._has_k5() or self._has_k33():
            return False
        return True

    def _has_k5(self):
        k5 = nx.complete_graph(5)
        return self._has_isomorphic_subgraph(k5)

    def _has_k33(self):
        k33 = nx.complete_bipartite_graph(3, 3)
        return self._has_isomorphic_subgraph(k33)

    def get_all_faces(self):
        def find_faces(node, parent, visited, path):
            if node in path:
                cycle = path[path.index(node):]
                if len(cycle) >= 3:
                    face_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
                    directed_edges = self.get_directed_edges()
                    undirected_edges = self.get_undirected_edges()
                    polarities = {}
                    for edge in face_edges:
                        for dedge in directed_edges:
                            if set(edge) == set(dedge):
                                if edge == dedge:
                                    polarities[edge] = 1
                                else:
                                    polarities[edge] = -1
                        for undedge in undirected_edges:
                            if set(edge) == set(undedge):
                                polarities[edge] = 0

                    face_dict = {}
                    for edge, polarity in polarities.items():
                        from_node, to_node = edge[0], edge[1]
                        if from_node not in face_dict:
                            face_dict[from_node] = {}
                        if to_node not in face_dict:
                            face_dict[to_node] = {}
                        face_dict[from_node][to_node] = polarity
                        face_dict[to_node][from_node] = -polarity

                    face_exists = False
                    for existing_face in faces:
                        if set(face_dict.keys()) == set(existing_face.keys()):
                            match = True
                            for key in face_dict.keys():
                                if set(face_dict[key].items()) != set(existing_face[key].items()):
                                    match = False
                                    break
                            if match:
                                face_exists = True
                                break

                    if not face_exists:
                        faces.append(face_dict)
                return
            path.append(node)
            visited.add(node)
            try:
                node_index = self.nodes.index(node)
                edge_indices = np.where(self.incidence_matrix[node_index] != 0)[0]
                neighbors = []
                for edge_index in edge_indices:
                    edge_nodes = np.where(self.incidence_matrix[:, edge_index] != 0)[0]
                    neighbors.extend([self.nodes[i] for i in edge_nodes if self.nodes[i] != node])
            except IndexError as e:
                raise
            for neighbor in neighbors:
                if neighbor != parent:
                    find_faces(neighbor, node, visited, path)
            path.pop()

        def find_exterior_face():
            G = self._to_networkx()
            planar_embedding = nx.planar_layout(G)
            exterior_face = set()
            for node, pos in planar_embedding.items():
                if pos[0] == min([p[0] for p in planar_embedding.values()]) or \
                pos[0] == max([p[0] for p in planar_embedding.values()]) or \
                pos[1] == min([p[1] for p in planar_embedding.values()]) or \
                pos[1] == max([p[1] for p in planar_embedding.values()]):
                    exterior_face.add(node)
            return list(exterior_face)

        faces = []
        visited = set()
        for node in self.nodes:
            if node not in visited:
                find_faces(node, None, visited, [])

        # Add the exterior face
        exterior_nodes = find_exterior_face()
        if exterior_nodes:
            exterior_face_dict = {node: {} for node in exterior_nodes}
            for i in range(len(exterior_nodes)):
                from_node = exterior_nodes[i]
                to_node = exterior_nodes[(i + 1) % len(exterior_nodes)]
                exterior_face_dict[from_node][to_node] = 0
                exterior_face_dict[to_node][from_node] = 0
            faces.append(exterior_face_dict)

        return faces

    def label_all_faces(self):
        faces = self.get_all_faces()
        print("Faces: ", faces)
    
        # Sort faces by the size of their keys
        faces = sorted(faces, key=lambda face: len(face.keys()), reverse=True)

        labeled_faces = {}
        face_counter = 0
        for face in faces:
            label = f"face_{face_counter}"
            labeled_faces[label] = face
            # invert face polarities and add as inverse
            inverse_face = {}
            for node, neighbors in face.items():
                inverse_face[node] = {}
                for neighbor, polarity in neighbors.items():
                    inverse_face[node][neighbor] = -polarity
            label = f"face_{face_counter}_inv"
            labeled_faces[label] = inverse_face
            face_counter += 1

        print("Labeled Faces: ", labeled_faces)
        self.faces = labeled_faces
        return labeled_faces
    
    def get_face_rings(self):
        # Sort faces into compatible rings.
        # Sort faces into tuples of faces and inverses
        # Faces with only undirected edges have no inverses
        faces = self.faces
        compatible_rings = []
        inverse_sets = {}
        for label in faces.keys():
            if '_inv' not in label:
                inverse_sets[label] = (faces[label], faces[label+'_inv'])
        print("Inverse sets: ", inverse_sets)

        visited = set()

        def face_to_frozenset(face):
            return frozenset((node, frozenset(neighbors.items())) for node, neighbors in face.items())

        def find_walk_between_compatible_faces(label, inverse=False):
            stack = [label]
            walk = []
            while stack:
                current_label = stack.pop()
                current_face = faces[current_label]
                current_face_frozen = face_to_frozenset(current_face)
                if current_face_frozen not in visited:
                    visited.add(current_face_frozen)
                    walk.append(current_label)
                    for next_label, (face, face_inv) in inverse_sets.items():
                        next_face = face if not inverse else face_inv
                        next_face_frozen = face_to_frozenset(next_face)
                        if next_face_frozen not in visited:
                            if self.check_face_compatibility(current_face, next_face) == 'Compatible':
                                stack.append(next_label if not inverse else f"{next_label}_inv")
            return walk

        # Find compatible walks for regular faces
        for label, (face, face_inv) in inverse_sets.items():
            if face_to_frozenset(face) not in visited:
                walk = find_walk_between_compatible_faces(label)
                if walk:
                    compatible_rings.append(walk)
            if face_to_frozenset(face_inv) not in visited:
                walk_inv = find_walk_between_compatible_faces(f"{label}_inv", inverse=True)
                if walk_inv:
                    compatible_rings.append(walk_inv)

        print("Compatible rings: ", len(compatible_rings))
        self.face_rings = compatible_rings
        return compatible_rings

    def get_ideal_ring(self):
        compatible_rings = self.get_face_rings()
        
        def count_inverses(ring):
            return sum(1 for label in ring if '_inv' in label)
        
        ideal_ring = min(compatible_rings, key=count_inverses)
        
        return ideal_ring

    def find_walk_between_faces(self, face_1, face_2):
        # Implement logic to find a walk between two compatible faces
        # This function should return a list representing the walk or None if no walk is found
        start_node = list(face_1.keys())[0]  # Starting from the first node in face_1
        target_node = list(face_2.keys())[0]  # Target is the first node in face_2
        visited = set()
        path = []

        def dfs(current_node):
            if current_node in visited:
                return False
            visited.add(current_node)
            path.append(current_node)
            if current_node == target_node:
                return True
            for neighbor in self.nodes:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            path.pop()
            return False

        if dfs(start_node):
            return path
        return None

    def check_face_compatibility(self, face_1, face_2):
        # check if there's any overlap of edges.
        # if so, check the polarity of the edges
        # if the polarities multiply to 1, the faces are compatible
        # if the polarities multiply to -1 the faces are incompatible
        # if the polarities multiply to 0, the faces are compatible
        overlap = set(face_1.keys()).intersection(set(face_2.keys()))
        if len(overlap) == 0:
            return 'Disjoint'
        elif face_1 != face_2:
            for node in overlap:
                for neighbor in face_1[node]:
                    if neighbor in face_2[node]:
                        if face_1[node][neighbor] == face_2[node][neighbor] == 0:
                            return 'Compatible'
                        elif face_1[node][neighbor]*face_2[node][neighbor] == -1:
                            return 'Incompatible'
                        elif face_1[node][neighbor]*face_2[node][neighbor] == 1:
                            return 'Compatible'
                    else:
                        return 'Disjoint'
        else:
            return 'Identical'
    
    def plot(self):
        G = self._to_networkx()

        # Use planar layout
        self.plotted_coordinates = nx.planar_layout(G)

        plt.figure(figsize=(12, 8))
        nx.draw(G, self.plotted_coordinates, with_labels=True, node_size=700, node_color='skyblue', font_size=16, font_color='darkblue')

        # Add arrows for directed edges
        ax = plt.gca()
        directed_edges = self.get_directed_edges()
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

    def get_faces(self):
        return self.faces

    def is_bipartite(self):
        nx_graph = self._to_networkx()
        return nx.is_bipartite(nx_graph)
    
    def get_shared_edges(self, face_1, face_2):
        shared_nodes = set(face_1.keys()).intersection(set(face_2.keys()))
        shared_edges = []
        for node in shared_nodes:
            for neighbor in face_1[node]:
                if neighbor in face_2[node]:
                    shared_edges.append((node, neighbor))
        return shared_edges

    def face_ring_to_dual_graph(self, face_ring=None):
        # Implement logic to convert a face ring to a dual graph
        # This function should return a PseudoDiGraph object
        if face_ring is None:
            face_ring = self.get_ideal_ring()
            print("No face ring provided, using estimated ideal ring: ", face_ring)

        faces = []
        diedges = []
        nondiedges = []
        for label in face_ring:
            faces.append(self.faces[label])
            face = self.faces[label]
            for label2 in face_ring:
                if label != label2:
                    shared_edges = self.get_shared_edges(self.faces[label], self.faces[label2])
                    for edge in shared_edges:
                        if face[edge[0]] == 1:
                            diedges.append((label, label2))
                        elif face[edge[0]] == -1:
                            diedges.append((label2, label))
                        else:
                            if (label, label2) not in nondiedges and (label2, label) not in nondiedges:
                                nondiedges.append((label, label2))

        print("Non-directed edges: ", nondiedges)
        print("Directed edges: ", diedges)
        
        pass

# Example usage
if __name__ == "__main__":

    graph = PlanarPDGraph()
    graph.add_edge(('A', 'B'))  # Directed edge from A to B
    graph.add_edge(('B', 'C'))  # Directed edge from B to C
    graph.add_edge(('C', 'A'))  # Directed edge from C to D
    graph.add_edge(('D', 'A'))  # Directed edge from D to A
    graph.add_edge(('B', 'D'))  # Directed edge from D to B

    print("Directed edges:", graph.get_directed_edges())
    graph.export(filename="graph_ordered.png")  # Saves the plot

    # Get faces of the graph
    faces = graph.label_all_faces()
    print("Faces and inverse faces in the graph:", len(faces))

    # Get face rings
    face_rings = graph.get_face_rings()
    print("Face rings: ", face_rings)

    # Get ideal ring
    ideal_ring = graph.get_ideal_ring()
    print("Ideal ring: ", ideal_ring)
    for label in ideal_ring:
        print(label, ": ", list(graph.faces[label].keys()))

    graph2 = PlanarPDGraph()
    graph2.add_edge(['A', 'B'])  # Directed edge from A to B
    graph2.add_edge(['B', 'C'])  # Directed edge from B to C
    graph2.add_edge(['C', 'A'])  # Undirected edge from C to A
    graph2.add_edge(['D', 'A'])  # Undirected edge from D to A
    graph2.add_edge(['B', 'D'])  # Directed edge from D to B
    graph2.add_edge(('F', 'D'))  # Directed edge
    graph2.add_edge(['F', 'E'])  # Directed edge
    graph2.add_edge(('E', 'D'))  # Directed

    print("Directed edges:", graph2.get_directed_edges())
    print("Undirected edges:", graph2.get_undirected_edges())
    graph2.export(filename="graph_unordered.png")  # Saves the plot

    # Get faces of the graph
    faces = graph2.label_all_faces()
    print("Faces and inverse faces in the graph:", len(faces))

    # Get face rings
    face_rings = graph2.get_face_rings()
    print("Face rings: ", face_rings)

    # Get ideal ring
    ideal_ring = graph2.get_ideal_ring()
    print("Ideal ring: ", ideal_ring)

    dual_graph = graph2.face_ring_to_dual_graph(ideal_ring)
    face0 = graph2.faces['face_0']  
    print("Face 0: ", face0)
    face3 = graph2.faces['face_3']
    print("Face 3: ", face3)
    comp = graph2.check_face_compatibility(face0, face3)
    print("Face compatibility: ", comp)

    #TODO Figure out why there's no external face including all of teh nodes from face_3

