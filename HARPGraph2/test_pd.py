from HARPGraph2.mixeddigraph import PseudoDiGraph
# Example usage
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
