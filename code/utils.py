# coding: utf-8
import numpy as np
import xgi
import matplotlib as mplt
import networkx as nx
from scipy.optimize import curve_fit
from scipy import stats

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def logistic(n, alpha, n0, L):
    return L/(1 + np.exp(-alpha*(n-n0)))

def PL_fit(ydata, xdata, xmin, xmax):
    
    def PL(x, K, beta):
        return K * x ** beta
    
    xmin, xmax = int(xmin), int(xmax)
    cutxdata=xdata[xmin:xmax]
    cutydata=ydata[xmin:xmax]
    popt, pcov = curve_fit(PL, cutxdata, cutydata)
    
    K, beta = popt[0], popt[1]
    yfit = list(map(lambda x: PL(x, K, beta), xdata))
    
    return beta, K, yfit

def PL_fit_fixed_beta(ydata, xdata, beta, xmin, xmax):
    
    def PL(x, K):
        return K * x ** beta
    
    xmin, xmax = int(xmin), int(xmax)
    cutxdata=xdata[xmin:xmax]
    cutydata=ydata[xmin:xmax]
    pars, pcov = curve_fit(PL, cutxdata, cutydata)
    
    K = pars[0]
    
    alpha = 0.05 # 95% confidence interval = 100*(1-alpha)
    n = len(ydata)    # number of data points
    p = len(pars) # number of parameters
    dof = max(0, n - p) # number of degrees of freedom
    # student-t value for the dof and confidence level
    tval = stats.distributions.t.ppf(1.0-alpha/2., dof) 
    
    Kmin, Kmax = [], []
    
    for i, p,var in zip(range(n), pars, np.diag(pcov)):
        sigma = var**0.5
        Kmin.append(p - sigma*tval)
        Kmax.append(p + sigma*tval)
        
    yfit = list(map(lambda x: PL(x, K), xdata))
        
    return K, yfit, Kmin, Kmax

def get_jaccard(x, y):
    """
    Returns Jaccard coefficient
    """
    x = set(x)
    y = set(y)
    return len(x.intersection(y))/len(x.union(y))

def normalize_dict(d, target=1.0):
    """
    Normalize the values of the input dictionary to the target value.
    """
    raw = sum(d.values())
    factor = target/raw

    return {key:value*factor for key,value in d.items()}

def get_Hs_from_groups_dict(groups_at_t_dict):
    """
    Convert a dictionary indexed by time and containing a list of groups at that time
    to a dictionary indexed by time and containing a xgi.Hypergraph() objects.
    """
    Hs={}
    
    for t, groups in groups_at_t_dict.items():
        H = xgi.Hypergraph()
        H.add_edges_from(groups)
        Hs[t] = H
        
    return Hs

def get_groups_dict_from_Hs(Hs):
    """
    Convert a list indexed by time and containing a xgi.Hypergraph() objects
    to a dictionary indexed by time and containing a list of group members at that time.
    """
    groups_at_t = {}
    
    if type(Hs)==list:
        for t, H in enumerate(Hs):
            groups_at_t[t]=list(H._edge.values())
    elif type(Hs)==dict:
        for t, H in Hs.items():
            groups_at_t[t]=list(H._edge.values())
    else:
        raise ValueError("Hs must be either a list or a dict")
        
    return groups_at_t

def get_cumulative_Gs_from_Hs(Hs):
    """
    Computes a cumulative graph of known nodes form a sequence
    of hypergraphs. Any time two nodes are part of a group a link
    is added to the growing graph (if not already present).
    
    Parameters
    ---------
    Hs: dict
        Input dictionary of xgi.Hypergraphs objects indexed by time.

    Returns
    -------
    Gs: dict
        Output dictionary of nx.Graph object indexed by time.

    """
    #Inizializing an empty dictionary where I will store the graphs
    Gs = {}
    #Empty graph where I will cumulatively store all interactions
    G = nx.Graph()
    #Adding nodes (useful to query before they have links)
    #G.add_nodes_from(set([n for H in Hs.values() for n in H.nodes]))
        
    #Scrolling over all Hypergraphs
    for t, H in Hs.items():
        #Extracting the 1-skeleton at this time
        G_t = xgi.convert.convert_to_graph(H)
        #Adding these links to the growing graph
        G.add_edges_from(G_t.edges)
        #Adding the graph to the list
        Gs[t] = G.copy()
    
    return Gs

def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def centered_np_hist(data, bins, density=True):
    """
    Same as numpy histogram but it returns bin centers instead.
    """
    density, bins = np.histogram(data, bins=bins, density=density)
    unity_density = density / density.sum()
    center = (bins[:-1] + bins[1:]) / 2
    
    return center, unity_density

def normalize_by_row(matrix):
    """
    Normalize numpy matrix by row.
    """
    row_sums = matrix.sum(axis=1)
    new_matrix = matrix / row_sums[:, np.newaxis]
    return new_matrix

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mplt.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def reduce_number_of_points(x,y,bins):
    bin_means, bin_edges, binnumber = stats.binned_statistic(x, y, statistic='mean', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    return bin_centers, bin_means