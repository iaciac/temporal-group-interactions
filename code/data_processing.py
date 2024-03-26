# coding: utf-8

import networkx as nx
import pandas as pd
import os

def LoadData(data_filename, data_dir):
  """
  Loads data from |data_dir|/|data_filename|. Returns Pandas dataframe.
  """
  return pd.read_csv(os.path.join(data_dir, data_filename))

def get_triadic_closure_links_to_add_at_time_t(df, timestamp):
    """
    Calculate the links that need to be added due to triadic closure to the CNS dataset.
    The input is the dataframe containing the interactions and the considered timestamp t.
    It returns a list of entries for the same input dataframe.
    """
    
    triclo_entries_to_add = []
    
    G = nx.Graph()
    
    # Interacting (id b >=0) users at given timestamp.
    df_t = df[(df["# timestamp"] == timestamp) & (df["user_b"] >= 0)][["user_a", "user_b", "rssi"]]
    
    # Building undirected network 
    for row in df_t.itertuples():
        user_a, user_b = row[1], row[2]
        rssi = row[3]
        G.add_edge(int(user_a), int(user_b), weight=rssi)
  
    #Iterating over nodes at time t
    for n in G.nodes:
        #If n has at least 2 neighbors
        if G.degree(n)>1:  
            neighbors = G[n]
            #Iterating over pairs of neighbors
            for i, j in combinations(neighbors, 2):
                #If there triangle is not already closed
                if not G.has_edge(i, j):
                    #I will need to close it, and I will assigne the MINIMUM RSSI to it
                    rssi = min(G.edges[n,i]['weight'], G.edges[n,j]['weight'])
                    triclo_entries_to_add.append({'# timestamp': timestamp,
                                                  'user_a': i,'user_b': j,
                                                  'rssi': rssi})

    return triclo_entries_to_add

def groups_at_time_t(df, timestamp, kind='cliques'):
    """
    Compute groups from the dataframe of temporal interactions.
    """
    G = nx.Graph()
    
    # Interacting (id b >=0) users at given timestamp.
    df_t = df[(df["# timestamp"] == timestamp) & (df["user_b"] >= 0)][["user_a", "user_b", "rssi"]]
    
    # Building undirected network 
    for row in df_t.itertuples():
        user_a, user_b = row[1], row[2]
        rssi = row[3]
        G.add_edge(user_a, user_b, weight=rssi)
  
    if kind == 'connected components':
        # Connected components
        connected_components = list((G.subgraph(c).copy() for c in nx.connected_components(G)))
        # From list of node views to list of lists
        groups = [list(cc.nodes) for cc in connected_components]
    elif kind == 'cliques':
        groups = list(nx.find_cliques(G))
    
    return groups
