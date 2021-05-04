import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


def energy(props, ed_enabled=True):
    prr = np.mean(props)
    n_ed, n_rx, n_tx = 3 if ed_enabled else 0, 7, 1
    i_ed, i_rx, i_tx = 5e-3, 5e-3, 10e-3
    t_ed, t_tx = 128e-6,  3.2e-3
    v_cc = 3.3
    return (i_ed * n_ed * t_ed + (i_rx * n_rx * t_tx + i_tx * n_tx * t_tx) / prr) * v_cc


if __name__ == '__main__':
    # Adjusting the DPI of the figures.
    mpl.rcParams.update({'figure.dpi': 300, 'font.size': 6})
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', choices=('apartments', 'downtown', 'suburb'), nargs='+')
    parser.add_argument('-alpha', type=float, default=0.95)
    args = parser.parse_args()
    # Data Preparation
    labels = ('TSCH', 'ETSCH', 'ITSCH (Mean BERs)', 'ITSCH (Max BERs)')
    columns = ('TSCH', 'ETSCH', 'ITSCH_Mean', 'ITSCH_Max')
    colors = ('gray', 'orange', 'green', 'blue')
    stored = {dataset: pd.read_csv('performances/{}.csv'.format(dataset)) for dataset in args.datasets}
    # PRPs
    width = 0.5
    # noinspection PyTypeChecker
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(3, 6))
    for count, (dataset, dataframe) in enumerate(stored.items()):
        for label, column, color in zip(labels, columns, colors):
            axes[count].plot(dataframe.Time, dataframe[column], label=label, color=color, linewidth=width)
        axes[count].set_xlim(dataframe.Time.min(), dataframe.Time.max())
        axes[count].set_title(dataset.capitalize())
        axes[count].set_ylabel('PRP')
    axes[count].set_xlabel('Time (Sec.)')
    plt.tight_layout()
    plt.show()
    # PRRs
    rows = list()
    for count, (dataset, dataframe) in enumerate(stored.items()):
        for column in columns:
            series = dataframe[column]
            n = series.size
            loc = np.mean(series)
            scale = scipy.stats.sem(series)
            start, end = scipy.stats.t.interval(args.alpha, n - 1, loc, scale)
            rows.append((dataset.capitalize(), column, loc, start, end))
    print(pd.DataFrame(rows, columns=('Scenario', 'Method', 'Mean', 'Interval Start', 'Interval End')))
    # Box Plots
    width = 0.75
    # noinspection PyTypeChecker
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(3, 6))
    positions = np.arange(len(labels)) + 1
    for count, (dataset, dataframe) in enumerate(stored.items()):
        axes[count].hlines(np.linspace(0, 1, 11), 0.25, 4.75, alpha=0.5, color='black', linewidth=0.1)
        sets = (dataframe.TSCH, dataframe.ETSCH, dataframe.ITSCH_Mean, dataframe.ITSCH_Max)
        plot = axes[count].boxplot(sets, positions=positions, widths=width, showfliers=False, patch_artist=True,
                                   meanprops={'color': 'black', 'linestyle': 'solid'}, meanline=True, showmeans=True,
                                   medianprops={'linewidth': 0})
        means = np.mean(sets, axis=1)
        axes[count].set_ylabel('PRP')
        axes[count].set_xlim(0.25, 4.75)
        axes[count].set_title(dataset.capitalize())
        axes[count].set_ylim(np.min(sets) * 0.95, np.max(sets) * 1.025)
        for patch, color in zip(plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    axes[count].set_xlabel('Protocol')
    plt.tight_layout()
    plt.show()
    # Energy Consumptions
    width = 1
    values = list()
    xticks, xticklabels = list(), list()
    figure, axes = plt.subplots(figsize=(3, 2))
    for count, (dataset, dataframe) in enumerate(stored.items()):
        ticks = np.arange(4) + count * 6
        xticks.append(np.median(ticks))
        xticklabels.append(dataset.capitalize())
        for tick, label, column, color in zip(ticks, labels, columns, colors):
            consumption = energy(dataframe[column]) * 1e6
            values.append(consumption)
            axes.bar(tick, consumption, label=label, color=color, width=width)
            axes.text(tick - width / 4, consumption + 10, '{:.2f}'.format(consumption), size='smaller', rotation='vertical')
    axes.set_xlabel('Scenario')
    axes.set_xticks(xticks)
    axes.set_xticklabels(xticklabels)
    axes.set_ylabel('Energy Consumption (μJ)')
    axes.set_ylim(np.min(values) * 0.8, np.max(values) * 1.10)
    plt.tight_layout()
    plt.show()


