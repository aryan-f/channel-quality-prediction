import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Adjusting the DPI of the figures.
    mpl.rcParams['figure.dpi'] = 300
    # Reading the computed metrics.
    results = pd.read_csv('results/summary.csv')
    # Plotting grouped bar charts.
    bar_width = 0.16
    scenarios = ('Apartments', 'Downtown', 'Suburb')
    protocols = ('TSCH', 'ETSCH', 'ITSCH (Mean BERs)', 'ITSCH (Max BERs)')
    colors = ('gray', 'orange', 'green', 'blue')
    # noinspection PyTypeChecker
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 5))
    for index, (scenario, protocol, prr, consumption) in results.iterrows():
        x = scenarios.index(scenario) + protocols.index(protocol) * bar_width
        axes[0].bar([x], [prr], bar_width, label=protocol.capitalize(), color=colors[protocols.index(protocol)])
        axes[0].text(x - bar_width / 3, prr + 0.02, prr, size='smaller', rotation='vertical')
        axes[1].bar([x], [consumption], bar_width, label=protocol.capitalize(), color=colors[protocols.index(protocol)])
        axes[1].text(x - bar_width / 3, consumption + 15, consumption, size='smaller', rotation='vertical')
    axes[0].set_ylim(0.55, 1.1)
    axes[0].set_ylabel('Packet Reception Ratio')
    axes[1].set_ylim(400, 800)
    axes[1].set_ylabel('Energy Consumption (μJ)')
    axes[1].set_xticks(np.arange(3) + 1.5 * bar_width)
    axes[1].set_xticklabels(scenarios)
    plt.show()
