import networkx as nx

def get_neighbors(G,node_id):
    neighbors = list(G.neighbors(node_id))
    return neighbors

def ring_graph(num_nodes):
    G = nx.cycle_graph(num_nodes)
    return G
