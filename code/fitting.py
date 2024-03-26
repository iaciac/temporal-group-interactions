# coding: utf-8
import numpy as np
from scipy.special import rel_entr
import pandas as pd

def compute_JSD(p, q, base=None, *, axis=0, keepdims=False):
    """
    Same as the scipy.spatial.distance.jensenshannon but fixing the
    problem that normalising zero vectors lead to NaNs.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    #Normalising, but only if it's not the full zero vector
    if np.any(p):
        p = p / np.sum(p, axis=axis, keepdims=True)
    if np.any(q):
        q = q / np.sum(q, axis=axis, keepdims=True)
    m = (p + q) / 2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum
    if base is not None:
        js /= np.log(base)
    return np.sqrt(js / 2.0)

def fit_gsize_dist(Pk_emp, pars_df, MODEL_IN_PATH, log=False):
    """
    Compute the Jensen-Shannon divergence (JSD) between the empirical group size distribution
    and the ones for the different model realization.

    Parameters
    ----------
    Pk_emp: pandas DataFrame
        The dataframe of the target group size distribution from empirical data
        with columns "k" and "Pk" for group size and its probability.
    pars_df: pandas DataFrame
        Dataframe that contains model parameters as columns. Indices are for different simulations.
    MODEL_IN_PATH: string
        Name of the root folder contaiing the folder of the precomputed group size distribution
        dataframes of the model indexed by the simulation id pars_id.

    Returns
    -------
    fit_df: pandas DataFrame
        The input pars_df with an additional column for the JSD. 
    """

    JSDs = {}

    for ID in pars_df.index:
        #Reading model gsize dist
        IN_FNAME = "Pk_pars_id%i.csv"%ID
        Pk_model = pd.read_csv(MODEL_IN_PATH+IN_FNAME)

        #Sort df by k
        Pk_model = Pk_model.sort_values(by='k')   

        #Merging model and data in a single df (adjusting for the length with 0s)
        Pk_mer = Pk_model.merge(Pk_emp, on='k', how='outer').fillna(0)

        if log:
            #JS distance on LOG -- fixing for zeros
            Pk_mer['Pk_x_log']=np.log10(np.array(Pk_mer['Pk_x']),
                                    out=np.zeros_like(np.array(Pk_mer['Pk_x'])),
                                    where=(np.array(Pk_mer['Pk_x'])!=0))
            Pk_mer['Pk_y_log']=np.log10(np.array(Pk_mer['Pk_y']),
                                    out=np.zeros_like(np.array(Pk_mer['Pk_y'])),
                                    where=(np.array(Pk_mer['Pk_y'])!=0))

            JSD = compute_JSD(Pk_mer['Pk_x_log'], Pk_mer['Pk_y_log'])
            JSDs[ID]=JSD
        else:
            JSD = compute_JSD(Pk_mer['Pk_x'], Pk_mer['Pk_y'])
            JSDs[ID]=JSD


    #Adding JSD column to dataframe
    fit_df = pars_df.copy()
    if log:
        fit_df['JSD_gsize_log']=pd.Series(JSDs)
    else:
        fit_df['JSD_gsize']=pd.Series(JSDs)
    
    return fit_df

def compute_JSD_trans_mat(df_emp_T, df_model_T, weighted=False, k_cut="min"):
    """
    Compute the Jensen-Shannon divergence (JSD) between two group transition matrices.

    Parameters
    ----------
    df_emp_T: pandas DataFrame
        The dataframe of a group transition matrix with columns "k(t)", "k(t+1)", and "Prob.".
    df_model_T: pandas DataFrame
        The dataframe of a group transition matrix with columns "k(t)", "k(t+1)", and "Prob.".
    weighted: bool (default=False)
        Optional parameter for adding decaying weights based on group size.
    k_cut: "min" or "max" (default="min")
        If the provided matrices are of different order the comparison is done for the
        minimum of maximum size among the two. 

    Returns
    -------
    The output JSD.
    """
    emp_max_k = df_emp_T['k(t)'].max()
    model_max_k = df_model_T['k(t)'].max()
    if k_cut=="min":
        max_k = min(emp_max_k, model_max_k)
    elif k_cut=="max":
        max_k = max(emp_max_k, model_max_k)
    else:
        raise ValueError("The parameter k_cut needs to be either 'min' or 'max'")
    
    JSDs = {}
    for k in range(1, max_k+1):
        df_emp_T_k = df_emp_T[df_emp_T['k(t)']==k][['k(t+1)','Prob.']]
        df_model_T_k = df_model_T[df_model_T['k(t)']==k][['k(t+1)','Prob.']]
        
        #Merging model and data in a single df (adjusting for the length with 0s)
        df_Pks = df_emp_T_k.merge(df_model_T_k, on='k(t+1)', how='outer').fillna(0)
        JSDs[k] = compute_JSD(df_Pks['Prob._x'], df_Pks['Prob._y'])
    
    if weighted:
        #Decaying function
        w = lambda x: 1 - 1/(max_k+1)*x
        return sum({w(k)*J for k, J in JSDs.items()})
        
    else:
        return sum(list(JSDs.values()))

def fit_gtrans_mat(T_emp, pars_df, MODEL_IN_PATH, weighted=False, k_cut=None):
    """
    Compute the Jensen-Shannon divergence (JSD) between the empirical group transition matrix
    and the ones for the different model realization.

    Parameters
    ----------
    T_emp: pandas DataFrame
        The dataframe of the target group transition matrix from empirical data
        with columns "k(t)", "k(t+1)", and "Prob.".
    pars_df: pandas DataFrame
        Dataframe that contains model parameters as columns. Indices are for different simulations.
    MODEL_IN_PATH: string
        Name of the root folder contaiing the folder of the precomputed group transition matrix
        dataframes of the model indexed by the simulation id pars_id.
    weighted: bool (default=False)
        Optional parameter for adding decaying weights to the JSD based on group size.
    k_cut: "min" or "max" (default="min")
        If the provided matrices are of different order the comparison is done for the
        minimum of maximum size among the two. 

    Returns
    -------
    fit_df: pandas DataFrame
        The input pars_df with an additional column for the JSD. 
    """
    
    JSDs = {}
    
    for ID in pars_df.index:
        #Reading model group transition matrix
        IN_FNAME = "T_pars_id%i.csv"%ID
        T_model = pd.read_csv(MODEL_IN_PATH+IN_FNAME)

        #Computing the JSD between the two matrices 
        JSDs[ID]=compute_JSD_trans_mat(T_emp, T_model, weighted=weighted, k_cut=k_cut)
        
    #Adding JSD column to dataframe
    fit_df = pars_df.copy()
    fit_df['JSD_T']=pd.Series(JSDs)

    return fit_df

def extract_matrix_vector(df, stat="average"):
    """
    Given the input dataframe for the group transition matrix
    with columns ['k(t)','k(t+1)','Prob.'], extracts a vector of
    mean values (or std) for k(t+1), weighted by 'Prob.', for each k(t).
    NOTE: for high k values it might return NaNs, if there are no entries.
    """
    def weighted_std(values, weights):
        """
        Return the weighted average and standard deviation.
        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        variance = np.average((values-average)**2, weights=weights)
        return np.sqrt(variance)

    if stat=="average":
        return df.groupby('k(t)').apply(lambda x: np.average(x['k(t+1)'], weights=x['Prob.']))
    elif stat=="std":
        return df.groupby('k(t)').apply(lambda x: weighted_std(x['k(t+1)'], weights=x['Prob.']))
    
def get_distance_from_vec(emp_vec, model_vec):
    """
    Compute the euclidean distance between two input vectors. 
    """
    #Joining the two vectors in a single dataframe
    agg = pd.concat([emp_vec, model_vec], axis=1)
    #There will be NaNs --for high k values. I will remove them if they are not both columns
    agg.dropna(axis='index', how='all', inplace=True)
    #Now the vectors have equal length, so I can substitute NaNs with zeros for a fair comparison
    agg.fillna(0, inplace=True)
    agg['diff'] = agg[0]-agg[1]
    agg['diff_squared'] = agg['diff']**2
    return np.sqrt(agg['diff_squared'].sum())