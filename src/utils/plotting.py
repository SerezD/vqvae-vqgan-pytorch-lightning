from matplotlib import pyplot as plt
import numpy as np


def close_all_plots():
    plt.close('all')


def codebook_usage_barplot(y):
    """
    :param y: heights for bar plot, numpy array.
    :return: the bar plot as plt
    """
    f = plt.figure()
    plt.bar(x=np.arange(0, len(y)), height=y)
    plt.xlabel('Codebook Indexes')
    plt.ylabel('Probability of Being chosen')
    plt.grid(True)
    return f
