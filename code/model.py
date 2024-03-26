# coding: utf-8
import xgi
import random
import networkx as nx
import copy
import numpy as np
import os

from utils import logistic, normalize_dict

def get_group_probability(known_to_n, group_members):
    """
    Compute the probability for n to join a group based on known nodes.
    """
    p = (1 + len(known_to_n.intersection(set(group_members))))/1+len(group_members)
    return p

def get_join_probabilities(known_to_n, groups_dict):
    """
    Given a set of known nodes and the current dictionary of groups,
    compute the dictionary of probability of joining each group [group_ID: prob].
    """
    probabilities={}
    for group_id, group_members in groups_dict.items():
        probabilities[group_id]=get_group_probability(known_to_n, group_members)

    #Normalising to make it a probability
    probabilities = normalize_dict(probabilities, target=1)

    return probabilities

class TemporalHypergraphModel():
    
        def __init__(self,
                     N=100,
                     t_max=1000,
                     beta=1,
                     alpha=0.5,
                     n0=10,
                     L=1,
                     epsilon=1,
                     verbose=False,
                     verbose_light=False):
            
            #Parameters
            
            self.N = N
            self.t_max = t_max
            self.beta = beta
            self.alpha = alpha
            self.n0 = n0
            self.L = L
            self.epsilon = epsilon
            self.verbose = verbose
            self.verbose_light = verbose_light
            
            #Variables
            
            #List of hypergraphs, the main output
            self.Hs = []
            #Tracking duration of group memberships from node POV with a dictionary node_ID: duration (# timesteps)
            self.residence_time = {node_ID: 1 for node_ID in list(range(self.N))}
            
            #I will use a graph to track interactions between nodes
            self.G = nx.Graph()
            #Adding nodes
            self.G.add_nodes_from(range(self.N))

            if self.verbose or self.verbose_light:
                print('Model initialised with default parameters.')
            
        def set_parameters(self, pars):
            """
            Parameters
            ----------
            N : integer
                Number of nodes
            t_max : integer
                Number of timesteps
            beta : float (default=0.5)
                The exponent of the residence time that is used to compute
                the probability for a node to leave the current group. 
            alpha : float (default=0.5)
                The logistic growth rate associated to the logistic function
                that controls the probability of joining a group of size n.
            n0 : integer (default=10)
                The n_0 midpoint value of the logistic function
                that controls the probability of joining a group of size n.
            L : integer (default=1)
                The numerator of the lofgistic function 
                that controls the probability of joining a group of size n.
            epsilon : integer (default=1)
                Number of empty groups that are available when a node decides
                to change group. This parameters thus controls for the
                willingness of a node to become isolated [dynamic probability].
            """
            for key in pars:
                setattr(self, key, pars[key])
                
            if self.verbose or self.verbose_light:
                print('Updated parameters:',
                      'N', self.N,
                      't_max', self.t_max,
                      'beta', self.beta,
                      'alpha', self.alpha,
                      'n0', self.n0,
                      'L', self.L,
                      'epsilon', self.epsilon)
                
        def reset(self):
            """
            Reset the model variables. It is usefull to call it after a change of parameters.
            """
            #List of hypergraphs, the main output
            self.Hs = []
            #Tracking duration of group memberships from node POV with a dictionary node_ID: duration (# timesteps)
            self.residence_time = {node_ID: 1 for node_ID in list(range(self.N))}
            
            #I will use a graph to track interactions between nodes
            self.G = nx.Graph()
            #Adding nodes
            self.G.add_nodes_from(range(self.N))
            
        def get_change_probability(self, n, H):
            """
            Compute the probability for node n of changing the  current group
            based on its residence time there.
            """
            #Computing the size of the current group of n
            sn = len(H.nodes.neighbors(n))+1   
            #The change-probability is given by
            p = logistic(sn, alpha=self.alpha, n0=self.n0, L=self.L)/(1.+self.residence_time[n]**self.beta/self.N)

            return p
        
        def change_group(self, n, H):
            """
            This function operatively performs the group change on the provided hypergraph H.
            """
            #This is the thresholded set of nodes known to n up to time t
            known_to_n = set([j for j in self.G.neighbors(n) if self.G[n][j]['weight']>=1])

            #Extracting the IDs of the current groups of n
            current_groups_of_n_IDs = H.nodes.memberships(n)

            if len(current_groups_of_n_IDs)>1:
                raise ValueError('Node %i belongs to more than one group!'%n)
            else:
                current_group_of_n_ID = next(iter(current_groups_of_n_IDs)) # {id}->id
                if self.verbose: print('Selected node %i belongs to group ID %i'%(n, current_group_of_n_ID))

            #GROUP CHOICE

            #Is n isolated?
            n_isolated = len(H.nodes.neighbors(n))==0

            #Current available groups to join at time t are the hyperedges...
            groups_dict = copy.copy(H._edge)
            #...but excluding the group than n is leaving...
            groups_dict.pop(current_group_of_n_ID)
            #Plus, if n is not already isolated, it could become one.
            if not n_isolated:
                #I will add epsilon empty groups to the one available for joining
                for i_eps in range(self.epsilon):
                    empty_group_ID = max(groups_dict.keys())+1
                    groups_dict[empty_group_ID]=[]
            if self.verbose: print("Node %i will choose among the available groups:", groups_dict)

            #Computing probabilies of joining groups
            probabilities = get_join_probabilities(known_to_n, groups_dict) 
            if self.verbose: print("Probabilities:", probabilities)
            #Choosing the group
            chosen_group_id = random.choices(list(probabilities.keys()), list(probabilities.values()))[0]
            if self.verbose: print("Chosen group ID:", chosen_group_id)

            #UPDATING OLD GROUP
            #I need to modify the group n was part of

            if n_isolated:
                #If the node was isolated I simply drop the group
                H.remove_edge(current_group_of_n_ID)
                if self.verbose:
                    print("Removing old isolated node...")
                    print(H._edge)
            else:
                #If the node was part of a group, I create a new one without the node
                past_group_members=list(H.nodes.neighbors(n)) #this one does not contain n, obviously
                H.remove_edge(current_group_of_n_ID)
                H.add_edge(past_group_members)
                if self.verbose:
                    print("Removing %i from the old group..."%n)
                    print(H._edge)

            #UPDATING NEW GROUP

            #Then, if n is becoming isolated (join empty group)
            if len(groups_dict[chosen_group_id])==0:
                #I create a new edge with just the node 
                H.add_edge([n])
                if self.verbose:
                    print("The node becomes isolated...")
                    print(H._edge)
            else:    
                #The node joins a 'real' group, whose members are
                chosen_group_members = groups_dict[chosen_group_id]
                #Size of the new group (including n)
                new_gsize = len(chosen_group_members)+1
                #So I first remove this new group
                H.remove_edge(chosen_group_id)
                if self.verbose:
                    print("Removing new group...")
                    print(H._edge)
                #And then I add id back but adding n
                H.add_edge(list(chosen_group_members)+[n])
                if self.verbose:
                    print("New group added")
                    print(H._edge)

                #At last, I update the graph that keeps track of connections 
                for m in chosen_group_members:

                    if self.G.has_edge(n, m):
                        #Updating the weight
                        self.G[n][m]['weight'] += 1/new_gsize
                    else:
                        #New edge
                        self.G.add_edge(n, m, weight=1/new_gsize)
                        
            return H

        def run(self):
            """
            Run the model with the specified parameters.
    
            Returns
            -------
            Hs : list
                List of xgi.Hypergraph objects

            """
            #Starting from an empty hypergraph
            H = xgi.Hypergraph()
            #Addig nodes (not really needed, would be created automatically from edges)
            H.add_nodes_from(range(self.N))
            shuffled_nodes=copy.deepcopy(list(H.nodes))
            #Adding also isolated nodes as edges (groups of size 1)
            H.add_edges_from([[i] for i in range(self.N)])
            self.Hs.append(H)

            ####### MAIN CYCLE OF THE MODEL ####### 
            for t in range(self.t_max):
                if self.verbose_light:
                    print('t=%i'%t)
                    print(list(H.edges.members()))
            
                #Shuffling the list of nodes at each iteration
                random.shuffle(shuffled_nodes)

                #For each node
                for n in shuffled_nodes:
                    if self.verbose_light: print('Selected node', n)

                    #Computing the probability for n to stay in the current group or change it
                    p = self.get_change_probability(n, H) 
                    if self.verbose_light: print('Probability of changing is', p)

                    #If n changes group
                    if random.random()<=p:
                        if self.verbose_light:
                            print('The agent will change group')
                        #Change
                        H = self.change_group(n, H) 
                        #Resetting the residence time of the node --since it changed group
                        self.residence_time[n]=1
                        if self.verbose_light:
                            print('The agent changed group')
                            print(list(H.edges.members()))
                    else:
                        #If n stays in the current group
                        if self.verbose_light:
                            print('The agent stays in the same group')
                            print(list(H.edges.members()))
                        #Incrementing its residence time in the current group
                        self.residence_time[n]+=1
                        pass

                self.Hs.append(H.copy())

            if self.verbose_light:
                print('End of simulation.')
                
            return self.Hs

###### SIMULATIONS - READ/WRITE ######

def run_from_df_and_save_edgelists(pars_id, pars_df, OUT_PATH):
    """
    Run a simulation based on the parameters in the provided dataframe and save.

    Parameters
    ----------
    pars_id: integer
        Index of the provided dataframe with the model parameters for the run.
    pars_df: pandas DataFrame
        Dataframe that contains model parameters as columns. Indices are for different simulations.
    OUT_PATH: string
        Results of the simulation will be saved in a folder inside the OUT_PATH folder,
        whose name is indexed by the simulation id pars_id.
        Edgelists at different times will be different csv files containing the time in the name. 
    """
    
    #Retrieving parameters
    pars_dict = pars_df.loc[pars_id].to_dict()
    #Fixing for wrong conversions of floats to ints by .to_dict() method
    pars_dict['N']=int(pars_dict['N'])
    pars_dict['t_max']=int(pars_dict['t_max'])
    pars_dict['epsilon']=int(pars_dict['epsilon'])
    
    #Running
    #print('Process', mp.current_process(), 'started...')
    Model = TemporalHypergraphModel()
    Model.set_parameters(pars_dict)
    Model.reset()
    print("ID %i - running the model..."%pars_id)
    Hs = Model.run()
    print("ID %i - end of simulation."%pars_id)
    
    #Saving
    DIR = OUT_PATH+"run_pars_id%i"%(pars_id)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    for t, H in enumerate(Hs):
        FNAME = "/yel_t%i.csv"%t
        xgi.write_edgelist(H, DIR+FNAME, delimiter=",")

def read_edgelists_from_df(pars_id, pars_df, IN_PATH):
    """
    Read the results of a single simulation with the provided parameters.

    Parameters
    ----------
    pars_id: integer
        Index of the provided dataframe with the model parameters for the already simulated run.
    pars_df: pandas DataFrame
        Dataframe that contains model parameters as columns. Indices are for different simulations.
    IN_PATH: string
        Name of the root folder contaiing the folder of the 
        performed simulation indexed by the simulation id pars_id.

    Returns
    -------
    Hs: list
        A list of xgi.Hypergraph object sorted by time.
    """
    
    Hs = []
    t_max = pars_df['t_max'][pars_id]
    DIR = IN_PATH+"run_pars_id%i"%(pars_id)
    
    for t in range(t_max):
        FNAME = "/yel_t%i.csv"%t
        H = xgi.read_edgelist(DIR+FNAME, delimiter=",", nodetype=int)
        Hs.append(H)
                
    return Hs