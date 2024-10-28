import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_radar(data, label, feature, filename, title=''):
    """ data = (N_label, N_feature) """
    cat = feature
    cat = [*cat, cat[0]]
    data = [[*d, d[0]] for d in data]

    ax = plt.subplot(polar=True)
    cat_loc = np.linspace(start=0, stop=2 * np.pi, num=len(cat))

    colors = ['#468dce', '#ffd044', '#d54141', '#ff8497', '#cb2fed']
    for idx, d in enumerate(data):
        ax.plot(cat_loc, d, 'o--', color=colors[idx], label=label[idx])
        ax.fill(cat_loc, d, alpha=0.15, color=colors[idx])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(cat_loc), cat)

    for xtick, angle in zip(ax.get_xticklabels(), cat_loc):
        if 0 < angle < np.pi:
            xtick.set_horizontalalignment('left')
        else:
            xtick.set_horizontalalignment('right')

    ax.set_ylim(0, 1.01)
    ax.set_rgrids([.2, .4, .6, .8, 1.])
    ax.invert_yaxis()

    ax.tick_params(axis='y', labelsize=8)
    ax.grid(color='#AAAAAA')
    ax.spines['polar'].set_color('#eaeaea')
    ax.set_facecolor('#FAFAFA')

    ax.set_title(title, y=1.08)
    if len(data) > 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig(filename)
    plt.show()
    plt.close()
