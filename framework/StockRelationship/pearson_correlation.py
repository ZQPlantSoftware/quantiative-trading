import matplotlib.pyplot as plt
import seaborn as sb

def pearson_correlation(stocks_num):
    '''
    Pearson Correlation of concept feature.
    In this case, minerals and elements from the periodic table.
    Output: (Run against the first 16 samples for this visualization example).
    It's also interesting to see how elements in the periodic table corelate to
    public companies. At some point, I'd like to use the data to predict breakthroughs
    a company might make based on their correlation to interesting elements or materials

    :param stocks: Training set
    :return:
    '''

    # f, ax = plt.subplots(figsize=(12, 10))
    plt.title('Pearson Correlation of Concept Features (Elements & Minerals)')

    # Draw the heatmap using seaborn (astype: convert)
    sb.heatmap(stocks_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
    plt.show()

