import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_results(input_list, title):

    # Collect keys:
    keys = np.unique([x[0] for x in input_list])
    features = ["Regressor"] + list(input_list[0][1][1].keys())
    values = np.array([[x[0]] + list(x[1][1].values()) for x in input_list])
    df = pd.DataFrame(values, columns=features)
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df = df.groupby("Regressor").mean()
    df = df.round(2)
    df = df.transpose()
    ax = df.plot(kind="bar", title=title, legend=True)
    ax.set_ylim([0, 1])

    plt.setp(plt.xticks()[1], rotation=0)

    rects = ax.patches

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.005, str(height), ha='center', va='bottom')

    #plt.legend(loc='best')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Regressors")

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.show()