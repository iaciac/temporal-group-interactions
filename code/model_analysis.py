# coding: utf-8
from collections import Counter, defaultdict
from itertools import chain
import numpy as np
import pandas as pd
import copy

from utils import get_jaccard, get_groups_dict_from_Hs, normalize_by_row

def group_size_dist(Hs):
    """
    Given a list of hypergraphs (output of the model),
    it computes the aggregated group size distribution.
    """
    sizes = list(chain(*[[order+1 for order in H.edges.order.aslist()] for H in Hs]))
    sizes_count = Counter(sizes)
    ks =  list(sizes_count.keys())
    Nks =  list(sizes_count.values())
    Pks = np.array(Nks)/sum(Nks)
    return ks, Pks

def normalize_transition_matrix(T):
    
    def normalize_by_row(matrix):
        row_sums = matrix.sum(axis=1)
        new_matrix = matrix / row_sums[:, np.newaxis]
        return new_matrix
    
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
    Hs: list
        Input list of xgi.Hypergraphs objects.
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
    timestamps = list(range(len(Hs)))
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
    Convert group transition matrix to pandas DataFrame.
    
    Parameter
    ---------
    T: numpy array
        Normalized group transition matrix.
        
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

def get_group_durations(Hs):
    """
    Compute group durations from list of hypergraphs.
    Each group size will be associated to a list of durations.

    Parameters
    ---------
    Hs: list
        Input list of xgi.Hypergraphs objects.
        
    Returns
    -------
    durations: defaultdict(list)
        Dictionary of lists. Each key is a group size. Each value is a list of group durations.
    """
    durations = defaultdict(list)

    #Converting the list of hypergraphs to a dictionary t: list of group members
    #I don't do it directly within the hypergraph because I will remove groups during the procedure
    groups_at_t = get_groups_dict_from_Hs(Hs)
    timestamps = groups_at_t.keys()

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

def get_group_times(Hs):
    """
    Compute group creation and destruction times from list of hypergraphs.

    Parameters
    ---------
    Hs: dict
        Input dict of xgi.Hypergraphs objects.
        
    Returns
    -------
    groups_and_times: list
        List of dictionaries. Each dictionary contains list of group members, creation, and destruction
        times of the group, that can be respectively accessed via "members", "t_start", and "t_end".
    """

    #I First convert the Hypergraph object into a dictionary of groups at time t
    groups_at_t_dict = get_groups_dict_from_Hs(Hs)

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

def get_dis_agg_matrices(Hs, groups_and_times, max_k, normed=True):
    """
    Compute group disaggregation and aggregation matrices.
    The (dis)aggregation matrix has rows that correspond to distributions of 
    the maximal size of the subset of the group that is (dis)aggregating.
    
    Parameters
    ---------
    Hs: dict
        Input dict of xgi.Hypergraphs objects.
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

    #I First convert the Hypergraph object into a dictionary of groups at time t
    groups_at_t_dict = get_groups_dict_from_Hs(Hs)

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

def get_full_dis_agg_matrices(Hs, groups_and_times, max_k, normed=True):
    """
    Compute group disaggregation and aggregation matrices.
    The (dis)aggregation matrix has rows that correspond to distributions of 
    the sizes (into) from which the group is (dis)aggregating.
    
    Parameters
    ---------
    Hs: dict
        Input dict of xgi.Hypergraphs objects.
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

    #I First convert the Hypergraph object into a dictionary of groups at time t
    groups_at_t_dict = get_groups_dict_from_Hs(Hs)

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

