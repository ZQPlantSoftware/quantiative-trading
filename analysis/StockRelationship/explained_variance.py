import numpy as np
import matplotlib.pyplot as plt

def explained_variance(stocks, X_std):
    '''
    Measuring Explained Variance
    Explained Variance = (Total variance - Residual variance)
    The number of PCA projection components that should be worth
    likking at can be guided by the Explained Variance Measure
    http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html
    :return:
    '''

    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse=True)

    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]

    # Cumulative explained variance
    cum_var_exp = np.cumsum(var_exp)

    # Variances plot
    max_cols = len(stocks.columns) - 1
    plt.figure(figsize=(10, 5))
    plt.bar(range(max_cols), var_exp, alpha=0.3333, align='center', label='individual explained variance', color='g')
    plt.step(range(max_cols), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.show()