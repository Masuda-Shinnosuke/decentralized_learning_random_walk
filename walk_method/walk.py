import numpy as np
from config.graph import get_neighbors

def simpleRandomwalk(graph,current_state):
    neighbors = get_neighbors(graph,current_state)
    next_state = np.random.choice(neighbors)
    return next_state
