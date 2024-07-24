import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_relation_bars(df: pd.DataFrame, ax=None, title=''):
    df_relation = df.relation.value_counts()
    df_relation_dist = df_relation / df_relation.sum()

    df_relation_dist.to_frame().plot(kind='bar', ax=ax)
    for i, n in enumerate(df_relation_dist):
        if ax:
            ax.text(i, n / 2, f'{np.round(n, 3)}%', va='center', ha='center')
            ax.set_title(title)
        else:
            plt.text(i, n / 2, f'{np.round(n, 3)}%', va='center', ha='center')
    return df_relation.reset_index()
