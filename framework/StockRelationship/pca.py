import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plt_pca(X_std):
    '''
    From this chart we can see that a large amount of variance
    comes from the first 85% of the predicted Principal Components
    It's a high number so let's start a low end and model for just
    a handful of Principal Component. More information on analyzing
    a reasonable number of Principal Components can be found:
    http://setosa.io/ev/principal-component-analysis/

    Using scikit-learn's PCA module, lets set n_components = 9
    The second line of the code calls the "fit_transform" methods,
    which fits the PCA model with the standardized movie data X_std
    and applies the dimensionality reduction on this dataset

    :param X_std:
    :return:
    '''
    pca = PCA(n_components=9)
    x_9d = pca.fit_transform(X_std)

    plt.figure(figsize=(9, 7))
    plt.scatter(x_9d[:, 0], x_9d[:, 1], c='goldenrod', alpha=0.5)
    plt.ylim(-10, 30)
    plt.show()

    return x_9d