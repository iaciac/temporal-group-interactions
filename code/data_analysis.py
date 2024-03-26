# coding: utf-8
import networkx as nx
from itertools import chain
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import copy
import random

from utils import get_jaccard, normalize_by_row


def groups_at_time_t(df, timestamp, dataset):
    """
    Compute groups at a given time from a dataframe of interactions.

    Parameters
    ----------
    df: pandas DataFrame
        Dataframe containing the dyadic interaction events and their time.
        Columns are ["# timestamp", "user_a", "user_b", "rssi"], the last only for the CNS data.
    timestamp: int
        Selected timestamp for the group interactions (it has to match a value of "# timestamp").
    dataset: either "CNS" or "DyLNet" or "Confs"

    Returns
    -------
    groups: List
        List of group interactions.
    """
    
    if dataset not in ["CNS", "DyLNet", "Confs"]:
        raise ValueError("dataset must be either 'CNS', 'DyLNet' or 'Confs'.")

    #Selecting the given timestamp. NOTICE I also have isolated nodes interacting with node_id -1
    if dataset=="CNS":
        df_t = df[(df["# timestamp"] == timestamp)][["user_a", "user_b", "rssi"]]
    else:
        df_t = df[(df["timestamp"] == timestamp)][["user_a", "user_b"]]
        
    #Building the undirected network 
    G = nx.Graph()

    for row in df_t.itertuples():
        user_a, user_b = row[1], row[2]
        if dataset=="CNS":
            rssi = row[3]
            G.add_edge(user_a, user_b, weight=rssi)
        else:
            G.add_edge(user_a, user_b)
  
    # Removing node -1 so that isolated nodes will actually remain isolated
    try:
        G.remove_node(-1)
    except:
        #There was no -1 node this time (in the DyLNet data)
        pass

    groups = list(nx.find_cliques(G))
    
    return groups

def group_size_dist(group_dict):
    """
    Given a dictionary of groups (from data), it computes the aggregated group size distribution.
    """
    sizes = list(chain(*[[len(group) for group in groups] for groups in group_dict.values()]))
    sizes_count = Counter(sizes)
    ks =  list(sizes_count.keys())
    Nks =  list(sizes_count.values())
    Pks = np.array(Nks)/sum(Nks)
    return ks, Pks

def normalize_transition_matrix(T):
        
    #Normalising by size of the target group (column index+1)
    norm_T = T/(np.arange(T.shape[-1])+1)

    #Normalizing matrix by row
    norm_matrix = normalize_by_row(norm_T)

    return norm_matrix

def get_transition_matrix(Hs, max_k = 20, normed=True):
    """
    Compute the node transition matrix from list of hypergraphs.
    Each entry is the number of times that a node belonging to a group of size k (x-axis)
    joins a different group of size k' (y-axis). The limit max_k is enforced, so if bigger
    groups appear they are not counted. 

    Parameters
    ---------
    Hs: dict
        Input dictionary of xgi.Hypergraphs objects, indexed by time of interaction.
    max_k: int
        Maximal dimension of the transition matrix to compute.
    normed: bool (default=True)
        The output matrix will be normalized by row.
        
    Returns
    -------
    T: np.array
        Transition matrix of shape (max_k, max_k).
    """
    
    #Initializing an empty transition matrix T to fill
    timestamps = list(Hs.keys())
    T = np.zeros((max_k, max_k), dtype=int)

    #Scrolling over all timestamps (except the last one)
    for t in timestamps[:-1]:
        #For each node present at time t
        for n in Hs[t].nodes:
            #If the node is also present at time t+1
            if n in Hs[t+1].nodes:
                #Looping over the groups of n at t
                for group_t in Hs[t].nodes.memberships(n):
                    #Computing the size of the group as the order of the edge+1
                    k_t = Hs[t].order(group_t)+1
                    #Looping over the groups of n at t+1
                    for group_t1 in Hs[t+1].nodes.memberships(n):
                        #Computing the size of the group as the order of the edge+1
                        k_t1 = Hs[t+1].order(group_t1)+1
                        
                        if k_t!=k_t1: #If there is a change in the group size
                            try:
                                T[k_t-1][k_t1-1]+=1       #-1 since it starts from 0
                            except IndexError: #One of the two group sizes is bigger than the provided max_k
                                continue
                        
                        else: #Same group size, but the groups might still be different
                            if get_jaccard(Hs[t]._edge[group_t], Hs[t+1]._edge[group_t1])!=k_t:
                                #Groups do not overlap perfectly, so it's OK
                                try:
                                    T[k_t-1][k_t1-1]+=1   #-1 since it starts from 0  
                                except IndexError: #One of the two group sizes is bigger than the provided max_k
                                    continue

    if normed:
        T = normalize_transition_matrix(T)

    return T

def transition_matrix_to_df(T):
    """
    Convert node transition matrix to pandas DataFrame.
    
    Parameter
    ---------
    T: numpy array
        Normalized node transition matrix.
        
    Returns
    -------
    df: pandas DataFrame
    """
    data_for_df = []
    
    for i in range(T.shape[0]):
        k=i+1 #i=0 is the index for group of size 1
        for j in range(T.shape[1]):
            k1=j+1
            prob = T[i,j]
            row=[k,k1,prob]
            data_for_df.append(row)

    df = pd.DataFrame(data_for_df, columns=['k(t)','k(t+1)','Prob.'])
    
    return df

def dis_agg_matrix_to_df(D):
    """
    Convert group (dis)aggregation matrix to pandas DataFrame.
    
    Parameter
    ---------
    D: numpy array
        Normalized group (dis)aggregation matrix.
        
    Returns
    -------
    df: pandas DataFrame
    """
    data_for_df = []
    
    for i in range(D.shape[0]):
        k=i
        for j in range(D.shape[1]):
            k1=j
            prob = D[i,j]
            row=[k,k1,prob]
            data_for_df.append(row)

    df = pd.DataFrame(data_for_df, columns=['k','k_prime','Prob.'])
    
    return df


def get_group_durations(groups_at_t_dict):
    """
    Compute group durations from dict of list group members indexed by time.
    Each group size will be associated to a list of durations.

    Parameters
    ---------
    groups_at_t_dict: dict
        Input dict of lists containing group members at a given time (key)
        
    Returns
    -------
    durations: defaultdict(list)
        Dictionary of lists. Each key is a group size. Each value is a list of group durations.
    """
    durations = defaultdict(list)

    #Converting the list of hypergraphs to a dictionary t: list of group members
    #I don't do it directly within the input because I will remove groups during the procedure
    groups_at_t = copy.copy(groups_at_t_dict)
    timestamps = list(groups_at_t.keys())

    for t in timestamps:
        for group in groups_at_t[t]:
            k=len(group)
            duration=1
            delta_t=1
            go_on=True
            while go_on:
                if (t+delta_t in timestamps) and (group in groups_at_t[t+delta_t]):
                    duration+=1
                    groups_at_t[t+delta_t].remove(group)
                    delta_t+=1
                else:
                    go_on=False
            durations[k].append(duration)
    return durations

def get_group_times(groups_at_t_dict):
    """
    Compute group creation and destruction times from dict of list group members indexed by time.

    Parameters
    ---------
    groups_at_t_dict: dict
        Input dict of lists containing group members at a given time (key).
        
    Returns
    -------
    groups_and_times: list
        List of dictionaries. Each dictionary contains list of group members, creation, and destruction
        times of the group, that can be respectively accessed via "members", "t_start", and "t_end".
    """

    #In this list I will put the dictionaries containing the information
    groups_and_times = []

    #Converting the list of hypergraphs to a dictionary t: list of group members
    #I don't do it directly within the input because I will remove groups during the procedure
    groups_at_t = copy.deepcopy(groups_at_t_dict)
    timestamps = list(groups_at_t.keys())

    for t in timestamps:
        for group in groups_at_t[t]:
            delta_t=1
            go_on=True
            while go_on:
                if (t+delta_t in timestamps) and (group in groups_at_t[t+delta_t]):
                    groups_at_t[t+delta_t].remove(group)
                    delta_t+=1
                else:
                    #the group terminated
                    go_on=False
                    #I can store the info of the group now
                    groups_and_times.append({'members': group, 't_start': t, 't_end': t+delta_t-1})
            
    return groups_and_times

def get_dis_agg_matrices(groups_at_t_dict, groups_and_times, max_k, normed=True):
    """
    Compute group disaggregation and aggregation matrices.
    The (dis)aggregation matrix has rows that correspond to distributions of 
    the maximal size of the subset of the group that is (dis)aggregating.
    
    Parameters
    ---------
    groups_at_t_dict: dict
        Input dict of lists containing group members at a given time (key)
    groups_and_times: list
        List of dictionaries. Each dictionary contains list of group members, creation, and destruction
        times of the group, that can be respectively accessed via "members", "t_start", and "t_end".
    max_k: int
        Maximal dimension of the transition matrix to compute.
    normed: bool (default=True)
        The output matrices will be normalized by row.
    
    Returns
    -------
    D: np.array
        Group disaggregation matrix of shape (max_k, max_k).
    A: np.array
        Group aggregation matrix of shape (max_k, max_k).
    """
    timestamps = list(groups_at_t_dict.keys())
    D = np.zeros((max_k, max_k))
    A = np.zeros((max_k, max_k))
    
    for event in groups_and_times:
        group = event['members']
        k_t = len(group)
        
        #DISAGGREGATION - I look after the group died
        next_t = event['t_end']+1
        
        #If after the end of the group we still have recordings
        if next_t in timestamps:
            #Groups present after the considered group finished
            all_next_groups = groups_at_t_dict[next_t]
            #I look at how the former members are divided into the new groups
            next_groups = [set(set(g).intersection(set(group))) for g in all_next_groups]
            #Removing empty intersections
            next_groups = list(filter(lambda g: len(g) > 0, next_groups))
            #If there is something left:
            if len(next_groups)>0:
                #Removing the case of growing group:
                #If all members are still together, I need to skip this group
                if set(group) in next_groups:
                    pass
                else:
                    #I loop over the next groups and I take only the one of biggest size
                    max_kt1 = max([len(g) for g in next_groups])
                    D[k_t][max_kt1]+=1
                    
        #AGGREGATION - I look before the group was born
        prev_t = event['t_start']-1
        
        #If before the beginning of the group we had recordings
        if prev_t in timestamps:
            #Groups present before the considered group started
            all_prev_groups = groups_at_t_dict[prev_t] 
            #I look at how the future members were divided into the old groups
            prev_groups = [set(set(g).intersection(set(group))) for g in all_prev_groups]
            #Removing empty intersections
            prev_groups = list(filter(lambda g: len(g) > 0, prev_groups))   
            #If there is something left:
            if len(prev_groups)>0:
                #Removing the case of shrinking group:
                #If all members were together in a bigger group, I need to skip this group
                if set(group) in prev_groups:
                    pass
                else:
                    #I loop over the previous groups and I take only the one of biggest size
                    max_kt1 = max([len(g) for g in prev_groups])
                    A[k_t][max_kt1]+=1
                    
    if normed:
        D = normalize_by_row(D)
        A = normalize_by_row(A)
    return D, A

def get_full_dis_agg_matrices(groups_at_t_dict, groups_and_times, max_k, normed=True):
    """
    Compute group disaggregation and aggregation matrices.
    The (dis)aggregation matrix has rows that correspond to distributions of 
    the sizes (into) from which the group is (dis)aggregating.
    
    Parameters
    ---------
    groups_at_t_dict: dict
        Input dict of lists containing group members at a given time (key)
    groups_and_times: list
        List of dictionaries. Each dictionary contains list of group members, creation, and destruction
        times of the group, that can be respectively accessed via "members", "t_start", and "t_end".
    max_k: int
        Maximal dimension of the transition matrix to compute.
    normed: bool (default=True)
        The output matrices will be normalized by row.
    
    Returns
    -------
    D: np.array
        Group disaggregation matrix of shape (max_k, max_k).
    A: np.array
        Group aggregation matrix of shape (max_k, max_k).
    """
    timestamps = list(groups_at_t_dict.keys())
    D = np.zeros((max_k, max_k))
    A = np.zeros((max_k, max_k))
    
    for event in groups_and_times:
        group = event['members']
        k_t = len(group)
        
        #DISAGGREGATION - I look after the group died
        next_t = event['t_end']+1
        
        #If after the end of the group we still have recordings
        if next_t in timestamps:
            #Groups present after the considered group finished
            all_next_groups = groups_at_t_dict[next_t]
            #I look at how the former members are divided into the new groups
            next_groups = [set(set(g).intersection(set(group))) for g in all_next_groups]
            #Removing empty intersections
            next_groups = list(filter(lambda g: len(g) > 0, next_groups))
            #If there is something left:
            if len(next_groups)>0:
                #Removing the case of growing group:
                #If all members are still together, I need to skip this group
                if set(group) in next_groups:
                    pass
                else:
                    #I loop over the next groups and I store all the sizes
                    for g in next_groups:
                        kt1 = len(g)
                        D[k_t][kt1]+=1
                    
        #AGGREGATION - I look before the group was born
        prev_t = event['t_start']-1
        
        #If before the beginning of the group we had recordings
        if prev_t in timestamps:
            #Groups present before the considered group started
            all_prev_groups = groups_at_t_dict[prev_t] 
            #I look at how the future members were divided into the old groups
            prev_groups = [set(set(g).intersection(set(group))) for g in all_prev_groups]
            #Removing empty intersections
            prev_groups = list(filter(lambda g: len(g) > 0, prev_groups))   
            #If there is something left:
            if len(prev_groups)>0:
                #Removing the case of shrinking group:
                #If all members were together in a bigger group, I need to skip this group
                if set(group) in prev_groups:
                    pass
                else:
                    #I loop over the previous groups and I store all the sizes
                    for g in prev_groups:
                        kt1 = len(g)
                        A[k_t][kt1]+=1
                    
    if normed:
        D = normalize_by_row(D)
        A = normalize_by_row(A)
    return D, A

def merge_dis_agg_matrices(D, A):
    """
    Merge two dataframes containing group and node disaggregation
    matrices into one. Notice that the disaggregation matrix becomes
    transposed.
    """
    #Keeping only lower triangular matrices
    D = D[D['k']>D['k_prime']]
    A = A[A['k']>A['k_prime']]
    #I will first take the 'transpost' of the D by swapping the columns
    D[['k','k_prime']]=D[['k_prime','k']];
    #Then I merge the dataframes by concatenation
    M = pd.concat([A, D], ignore_index=True)
    #Pivoting
    M = pd.pivot(data=M, index='k', columns='k_prime', values='Prob.')
    
    return M

def parse_dis_agg_matrices(D, A, align=False):
    """
    Parsing two dataframes containing group aggregation and disaggregation matrices.
    The disaggregation matrix gets moved to cover the upper diagonal. 
    The two input matrices are squared, the output are rectangular. 
    If align is true, the the "diagonal" will become a vertical line
    so that group sizes are aligned. In this case the output layout
    will look like a tree.
    """
    #Dropping zeros
    D = D[~((D['k']==0)|(D['k_prime']==0))]
    A = A[~((A['k']==0)|(A['k_prime']==0))]
    
    #I flip the D matrix horizontally along the diagonal
    D['k_prime']=D['k']+(D['k']-D['k_prime'])
    #This creates a negative values on the first row, which I delete
    D = D[D['k_prime']>0]
    if not align:
        #It also makes the matrix not squared anymore,
        #so I need to add empty columns to A as well to match the size
        old_max = A['k_prime'].max()
        new_max = (old_max*2)-1
        phantom_ks = list(range(old_max+1, new_max+1))
        phantom_rows = pd.DataFrame({"k":[old_max]*len(phantom_ks),
                                     "k_prime":phantom_ks,
                                     "Prob.":[np.nan]*len(phantom_ks)})
        A = pd.concat([A, phantom_rows])    
    
    #Aligning the layout so that columns sizes match
    if align:
        A_max = A['k'].max()
        D_max = D['k'].max()
        D['k_prime'] = D['k_prime']+D_max-D['k']
        A['k_prime'] = A['k_prime']+A_max-A['k']
        
        #I need to add Nans to the disaggregation matrix since I
        #pushed everything to the right (before pivoting)
        k_prime_min = D['k_prime'].min()
        phantom_k_primes = list(range(1, k_prime_min))
        phantom_rows = pd.DataFrame({"k":[k_prime_min]*len(phantom_k_primes),
                                     "k_prime":phantom_k_primes,
                                     "Prob.":[np.nan]*len(phantom_k_primes)})
        D = pd.concat([D, phantom_rows])    
        
    #Pivoting
    D = pd.pivot(data=D, index='k', columns='k_prime', values='Prob.')
    A = pd.pivot(data=A, index='k', columns='k_prime', values='Prob.')
    
    #Replacing zeros with NaNs
    D = D.replace(0, np.nan)
    A = A.replace(0, np.nan)
    
    return D, A

def get_avg_curve_dis_agg_matrix(D):
    """
    Given a parsed (dis)aggregation matrix, already pivoted to be ready
    to plot as a heatmap, return the weighted average curve.
    """
    #List of y values
    y = [i for i in D.index]
    #Here I will put the average x values weighted by the prob.
    x = []
    for row_index in y:
        #Selecting one row at the time
        values = D.loc[row_index].index
        weights = np.array(D.loc[row_index].values)
        weights[np.isnan(weights)] = 0
        try:
            mean = np.average(values, weights=weights)
        except ZeroDivisionError:
            mean = np.nan
        x.append(mean)
    #I shift everyting so that values are placed at the center of the cells
    x = np.array(x)-0.5
    y = np.array(y)-0.5
    
    return x, y

def get_group_similarity(Hs, normed=False):
    """
    Compute the group Jaccard similarity.
    Given a series of hypergraphs, compute for each pair of consecutive time steps
    the overlap of consecutive groups of each node (Jaccard similarity).
    Each entry is a list of similarity values indexed by a pair of group sizes
    at consecutive times.

    Parameters
    ---------
    Hs: dict
        Input dictionary of xgi.Hypergraphs objects indexed by time of interaction.
        
    Returns
    -------
    J: defaultdict
       Dictionary of group similarity at consecutive times.
       Indexed by (k(t), k(t+1)), while values are lists of Jaccard similarity values.
    """

    
    timestamps = list(Hs.keys())
    
    #Here I will store the Jaccard values, keys will be the tuple (k,k')
    J = defaultdict(list)
    
    #Scrolling over all timestamps (except the last one)
    for t in timestamps[:-1]:
        #For each node present at time t
        for n in Hs[t].nodes:
            #If the node is also present at time t+1 (and t+1 exists)
            try:
                if n in Hs[t+1].nodes:
                    #Looping over the groups of n at t
                    for group_t in Hs[t].nodes.memberships(n):
                        #Computing the size of the group as the order of the edge+1
                        k_t = Hs[t].order(group_t)+1
                        #Looping over the groups of n at t+1
                        for group_t1 in Hs[t+1].nodes.memberships(n):
                            #Computing the size of the group as the order of the edge+1
                            k_t1 = Hs[t+1].order(group_t1)+1

                            #Storing the Jaccard coefficient
                            if not normed:
                                J[(k_t, k_t1)].append(get_jaccard(Hs[t]._edge[group_t], Hs[t+1]._edge[group_t1]))
                            else:
                                J[(k_t, k_t1)].append(get_jaccard(Hs[t]._edge[group_t], Hs[t+1]._edge[group_t1])/min(k_t, k_t1))
            except KeyError:
                pass
    return J

def get_avg_group_similarity(J, remove_ones=False):
    """
    Given the full lists of group similarity at consecutive timesteps
    (output of the function 'get_group_similarity')
    for every pair of group sizes k, k', compute the average similarity.
    
    Parameter
    ---------
    J: defaultdict
       Dictionary of group similarity at consecutive times.
       Indexed by (k(t), k(t+1)), while values are lists of Jaccard similarity values.
    remove_ones: bool
       If true, perfect overlaps (similarity=1) are discarded before computing the statistics.
        
    Returns
    -------
    df: pandas DataFrame
        Average similarity for each pair k, k'
    """
    data_for_df = []
    
    for k_pair, values in J.items():
        k = k_pair[0]
        k1 = k_pair[1]
        if remove_ones:
            clean_values = [v for v in values if v != 1]
        else:
            clean_values = values
        #Checking that the removal didn't produce an empty list
        if len(clean_values)>0:
            mean_J = np.nanmean(clean_values)
            median_J = np.nanmedian(clean_values)
            row=[k,k1,mean_J, median_J]
            data_for_df.append(row)
        else:
            pass
        
    df = pd.DataFrame(data_for_df, columns=['k(t)','k(t+1)','mean J', 'median J'])
    
    return df

def get_probs_leaving_group(durations, gsizes, taus=np.arange(1,1000)):
    """
    Compute the probability of leaving a group of a given size after
    a given time.
    
    Parameters
    ---------
    durations: defaultdict(list) 
        Output of the function get_group_durations()
        It is a Dictionary of lists: each key is a group size, each value is a list of group durations.
    gsizes: list
        List of group sizes to use for the computation.
    taus: list (default np.arange(1,1000))
        List of group durations to use for the computation.
    
    Returns
    -------
    prob_by_size: dict
        Dictionary with key group size and values the list of probabilities as a function of taus.
    """
    #Initiating the dictionary that I will return
    prob_by_size = {}
    
    for k in gsizes:
        probs=[]
        for tau in taus:
            num_groups_at_least_tau = sum(dur>=tau for dur in durations[k])
            num_nodes_at_least_tau = num_groups_at_least_tau*k
            
            num_groups_tau = sum(dur==tau for dur in durations[k])
            num_nodes_tau = num_groups_tau*k
            
            if num_nodes_at_least_tau!=0:
                prob_tau = num_nodes_tau/num_nodes_at_least_tau 
                probs.append(prob_tau)
            else:
                probs.append(np.nan)
                
        prob_by_size[k]=probs
        
    return prob_by_size

def measure_social_memory(Hs, groups_at_t_dict, Gs, groups_and_times):
    """
    Measuring social memory in the system by comparing the density of
    known nodes after a group change vs the random case. 
    
    Parameters
    ----------
    Hs: dict
        Input dictionary of xgi.Hypergraphs objects indexed by time.
        
    groups_at_t_dict: dict
        Input dict of lists containing group members at a given time (key)
        
    Gs: dict
        Dictionary of nx.Graph object indexed by time containing the cumulative networks of contacts.

    groups_and_times: list
        List of dictionaries. Each dictionary contains list of group members, creation, and destruction
        times of the group, that can be respectively accessed via "members", "t_start", and "t_end".

    Returns
    -------
    df: pandas DataFrame
        Dataframe that contains time of change,
        density of nodes in chosen and random groups and
        the associated group size.
    """

    #I will store the results in a dataframe
    data_for_df = []
    
    timestamps = list(Hs.keys())

    for event in groups_and_times:
        group = event['members']

        #Extracting the cumulative graph of nodes that ever met at t
        G = Gs[event['t_end']]

        #I look after the group vanishes
        next_t = event['t_end']+1

        #If after the end of the group we still have recordings (within the selected context)
        if next_t in timestamps:            
            #Looping over the nodes of the considered group that dies at t
            for n in group:
                #I need to take all the groups present after the considered group finished
                #Check for patological case:
                #From this ones I need to remove, if any, those that include n that existed already at t

                gt = groups_at_t_dict[event['t_end']] #all groups at t
                gt1 = groups_at_t_dict[next_t] #all groups at t+1
                #Converting to sets and selecting only those that have n
                gnt_set = set([tuple(g) for g in gt if n in g])
                gnt1_set = set([tuple(g) for g in gt1 if n in g])
                #Groups with n present at both t and t+1
                gntt1_set = gnt_set.intersection(gnt1_set)
                #I now remove these last groups from the groups at t+1
                gt1_set = set([tuple(g) for g in gt1]) #all groups at t+1 (set)
                good_gt1_set = gt1_set.difference(gntt1_set) #Removing the persistent ones

                #Finally converting back to list
                all_next_groups = list([list(g) for g in good_gt1_set])
                #Dropping pointless variables
                del gt, gt1, gnt_set, gnt1_set, gntt1_set, gt1_set, good_gt1_set
                
                #I now consider the groups of n at t+1 --if n is not isolated
                next_groups = [g for g in all_next_groups if (n in g) and (len(g)>1)]
                
                #If there's any
                if len(next_groups)>0:
                    #Iterating over all the groups of n at t+1 (rarely might be more than one)
                    for next_group in next_groups:
                        
                        #######################################
                        # 'Real' social memory
                        #######################################
                        #The actual density from transitions found in data
                        
                        size_chosen_group = len(next_group)
                        #Computing the number of nodes of this chosen group already met at t
                        try:
                            num_known_nodes = len([j for j in G.neighbors(n) if j in next_group])
                        except:
                            #if n only appeared isolated, it won't have neighbors, so there would be an error
                            num_known_nodes = 0
                        #Computing the density by normalising with group size (removing n)
                        density_known_nodes = num_known_nodes/(size_chosen_group-1)
                        
                        #######################################
                        # Null model 1
                        #######################################
                        #As a baseline, I also compute the density with respect to a random group
                        
                        #I look for a random group available that is not the pathological case of node n isolated
                        trials_count = 0
                        max_trials_allowed = 30
                        found = False
                        
                        while (not found) and (trials_count < max_trials_allowed):
                            random_group = random.choice(all_next_groups)   
                            if not ((n in random_group) and len(random_group)==1):
                                #Found a good random group
                                found = True
                            else:
                                #I'll look for another random group
                                trials_count+=1

                        #If I found a good random group
                        if found==True:    
                            #Computing the number of nodes of this random group already met at t
                            try:
                                num_known_nodes_random = len([j for j in G.neighbors(n) if j in random_group])
                            except:
                                #if n only appeared isolated, it won't have neighbors, so there would be an error
                                num_known_nodes_random = 0
                            #Computing normalisation (excluding the case of n isolated)
                            if n in random_group:
                                norm_random = len(random_group)-1
                            else:
                                norm_random = len(random_group)
                            #Computing the density by normalising with group size
                            density_known_nodes_random = num_known_nodes_random/norm_random
                            size_random_group = len(random_group)
                        else:
                            density_known_nodes_random = np.nan
                            size_random_group = np.nan
                            
                        #######################################
                        # Null model 2
                        #######################################
                        #Another baseline, a random group of the same size of the 'real' chosen one
                        
                        #The target size is the size of the chosen group after removing node n
                        target_size = size_chosen_group-1
                        
                        if target_size==0:
                            #It was the case of n getting isolated, so
                            density_known_nodes_random_given_size = np.nan
                        else:
                            #I calculate the groups at t+1 of the target size but excluding n
                            next_groups_given_size = [g for g in all_next_groups if (len(g)==(target_size)) and (n not in g)] 
                        
                            #If there is at least one group of the desiderd size:
                            if len(next_groups_given_size)>1:
                                #I pick a random group
                                random_group_given_size_draft = random.choice(next_groups_given_size)
                                #I compute the number of nodes of this random group that n already met up to t
                                try:
                                    num_known_nodes_random_given_size = len([j for j in G.neighbors(n) if j in random_group_given_size])
                                except:
                                    #if n only appeared isolated, it won't have neighbors, so there would be an error
                                    num_known_nodes_random_given_size = 0
                                #Computing the density by normalising with group size
                                density_known_nodes_random_given_size = num_known_nodes_random_given_size/target_size
                            else:
                                #If there are no groups of the desired size left, the only one was the chosen one
                                density_known_nodes_random_given_size = density_known_nodes

                        #######################################
                        #Appending the computed densities
                        row = [event['t_end'], density_known_nodes,
                               density_known_nodes_random, size_chosen_group,
                               size_random_group, density_known_nodes_random_given_size]
                        data_for_df.append(row)
                else:
                    #node n was not present at t+1, nothing happens
                    pass

    df = pd.DataFrame(data_for_df, columns=['time', 'density_known_nodes_chosen_group',
                                            'density_known_nodes_random_group',
                                            'size_chosen_group', 'size_random_group',
                                            'density_known_nodes_random_group_given_size'])

    return df

def get_interevent_times(groups_and_times):
    """
    Measure interevent times between consecutive occurrences of the same group
    aggregated at the level of group size.
    
    Parameters
    ----------        
    groups_and_times: list
        List of dictionaries. Each dictionary contains list of group members, creation, and destruction
        times of the group, that can be respectively accessed via "members", "t_start", and "t_end".
        
    Returns
    -------
    interevent_dict: Defaultdict
        Defaultdict of lists with group sizes as keys and lists of times as lists of values.
    """

    def add_interevent_time_to_df(df, start, end):
        """
        Computes inter-event time in a dataframe by subtracting the start time of a row with the 
        end of the previous row. Each interevent time is start(t) - end(t-1)
        """
        # Compute the difference between column start (t) and shifted column end (t-1)
        df['interevent_t'] = df[start] - df[end].shift(1)
        return df
    
    #Converting groups and times into a pandas dataframe
    df = pd.DataFrame.from_dict(groups_and_times)
    #Converting the lists of group members into tuples
    df["members"] = df["members"].apply(lambda x: tuple(x))

    interevent_dict = defaultdict(list)
    
    for group, df_group in df.groupby("members"):
        #Size of the group
        g_size = len(group)
        #Computing the interevent times for this group from the dataframe
        df_int = add_interevent_time_to_df(df_group, "t_start", "t_end")
        #Extracting the list of interevent times
        interevent_list = list(df_int["interevent_t"].dropna())    
        #Adding the times to the dictionary with group size keys
        interevent_dict[g_size].extend(interevent_list)
        
    return interevent_dict

def get_node_trajectory(Hs):
    """
    Given a sequence of empirical hypergraphs, (N,T) matrix indexed by
    node and time with elements represening the size of the group the
    nodes belong to.
    
    Parameters
    ---------
    Hs: dict
        Input dictionary of xgi.Hypergraphs objects indexed by time of interaction.
    
    Returns
    -------
    A: np.array
        Matrix where:
            rows are rescaled node labels;
            columns are times (as given by the keys of the provided dictionary, but rescaled so that it starts from 0);
            vales are size of the group
    index_to_node: dict
        Dictionary to map back the indexes of the output matrix to the original node labels

    """
    
    #All nodes that appear at any time
    nodes = list({n for t in Hs.keys() for n in Hs[t].nodes})
    N = len(nodes)
    
    #Nodes are not necessarily labeled from 0 to N-1, so I need to relabel
    node_to_index = {n:i for i, n in enumerate(nodes)}
    index_to_node = {i:n for i, n in enumerate(nodes)}
        
    #Times do not start from 0, so I will have to rescale
    t_min = min(Hs.keys())
    t_max = max(Hs.keys())
    T = t_max-t_min
    
    #Creating an empty array that I will fill. By default nodes are isolated. 
    A = np.ones((N, T+1))

    for t, H in Hs.items():
        for group in H.edges.members():
            for n in group:
                A[node_to_index[n], t-t_min] = len(group)
                
    return A, index_to_node