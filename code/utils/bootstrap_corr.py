import numpy as np
import scipy.stats as stats

def bootstrap_corr(a,b,nbootstraps=1000,ci=95,corr_type='pearson'):
    """
    compute confidence intervals for pairwise correlations
    """
    n_units = a.shape[0]
    ind = np.arange(n_units)
    bootstrapped = []
    for i in range(nbootstraps):
        bootstrap_ind = np.random.choice(ind,size=n_units,replace=True)
        if corr_type=='pearson':
            r = stats.pearsonr(a[bootstrap_ind],b[bootstrap_ind])[0]
        if corr_type=='spearman':
            r = stats.spearmanr(a[bootstrap_ind],b[bootstrap_ind])[0]
        bootstrapped.append(r)
    
    # calculate CI
    lowerci = (100-ci)/2.0
    upperci = ci + lowerci
    sorted_vals = np.sort(bootstrapped)
    lower_ind = int(nbootstraps*(lowerci/100))
    upper_ind = int(nbootstraps*(upperci/100))
    lowerbound = sorted_vals[lower_ind]
    upperbound = sorted_vals[upper_ind]
    
    if corr_type=='pearson':
        r = stats.pearsonr(a,b)[0]
    if corr_type=='spearman':
        r = stats.spearmanr(a,b)[0]
    return r, lowerbound, upperbound
