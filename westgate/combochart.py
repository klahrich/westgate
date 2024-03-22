import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def combo_chart(df, xvar, yvar, bins=None, q=None, ylabel=None, savefile=None):
    df = df.copy()

    xtitle = xvar

    if np.issubdtype(df[xvar], np.number):
        if bins is not None:
            df['bins'] = pd.cut(df[xvar], 
                                bins)
        elif q is not None:
            df['bins'] = pd.qcut(df[xvar], q, duplicates = 'drop')

        xvar = 'bins'
    
    hist = df[xvar].value_counts(sort=False)
    dfr = df.groupby(xvar)[yvar].mean()

    df_plt = pd.DataFrame({
        xvar: hist.index.values.astype(str),
        'count': hist,
        yvar: dfr
    })

    plt.rc('xtick', labelsize=6) 
    
    fig, ax1 = plt.subplots()
    ax1.tick_params(axis='x', labelrotation=45)

    ax1.bar(df_plt[xvar], df_plt['count'], color='orange', alpha=0.5)
    ax1.legend(['Count'], loc="upper left")

    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(df_plt[xvar], df_plt[yvar], linestyle='-', marker='o')
    ax2.legend([ylabel if ylabel else yvar], loc="upper right")

    if savefile:
        plt.savefig(savefile)
        
    return fig


def combo_chart_plotly(df, xvar, yvar, bins=None, q=None, ylabel=None):
    df = df.copy()

    xtitle = xvar

    if np.issubdtype(df[xvar], np.number):
        if bins is not None:
            df['bins'] = pd.cut(df[xvar], 
                                bins)
        elif q is not None:
            df['bins'] = pd.qcut(df[xvar], q, duplicates = 'drop')

        xvar = 'bins'
    
    hist = df[xvar].value_counts(sort=False)
    dfr = df.groupby(xvar)[yvar].mean()

    df_plt = pd.DataFrame({
        xvar: hist.index.values.astype(str),
        'count': hist,
        yvar: dfr
    })

    fig = make_subplots(
        subplot_titles=[xvar],
        specs=[[{"secondary_y": True}]]
    )

    fig.add_trace(
        go.Bar(
            x = df_plt[xvar],
            y = df_plt['count'],
            name = "Count"
        )
    )

    fig.add_trace(
        go.Scatter(
            x = df_plt[xvar],
            y = df_plt[yvar],
            name = ylabel if ylabel else yvar,
            mode='lines+markers'
        ),
        secondary_y=True
    )

    fig.update_layout(
        title_text=xtitle
    )

    # Set x-axis title
    #fig.update_xaxes(title_text="xaxis title")

    # Set y-axes titles
    fig.update_yaxes(title_text='volume', secondary_y=False)
    fig.update_yaxes(title_text=yvar, secondary_y=True)

    return fig