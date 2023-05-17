import numpy as np
import matplotlib
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes', labelsize=15)
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 10000
from soccer_game import SoccerGame


def plot_error(error, name):
    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    plt.plot(error, color='black', linestyle='-')
    plt.ylim(0, 0.5)
    plt.title(name)
    plt.xlabel("Iteration")
    plt.ylabel("Q value difference")
    fig.savefig(name + ".png", dpi=fig.dpi)
    return