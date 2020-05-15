import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from protocols import tsch, enhanced_tsch, intelligent_tsch
from model import Network
from simulation import BitErrorProb, PacketReceptionProb


def reduce(values, method, matrix=None):
    if method == 'mean':
        channels_sums = torch.sum(values, dim=0)
        channels_weights = torch.sum(matrix, dim=0)
        return channels_sums / channels_weights
    values, indices = torch.max(values, dim=0)
    return values


def describe(receptions, title):
    print(title)
    prr = torch.mean(receptions)
    n_ed, n_rx, n_tx = 3, 7, 1
    i_ed, i_rx, i_tx = 5e-3, 5e-3, 10e-3
    t_ed, t_tx = 128e-6,  3.2e-3
    v_cc = 3.3
    energy = (i_ed * n_ed * t_ed + (i_rx * n_rx * t_tx + i_tx * n_tx * t_tx) / prr) * v_cc
    print('PRR: {:.4f}, Energy Consumption: {:.2f} uJ'.format(prr, energy * 1e6))


if __name__ == '__main__':
    # Configuration
    dataset = 'apartments'
    sample_rate, device_sample_rate = 2000, 10
    power, alpha, distance = -10., 3.5, 3.
    packet_length = 133  # Bytes
    past_window, future_window = 5, 5  # Seconds
    layers, neurons = 2, 32
    iterations, batch_size = 1000, 50
    # Dataset Preparation
    data = torch.load(f'data/{dataset}.pt')
    datapoints, sequences, channels = data.shape
    training_data = data[:int(0.80 * datapoints)]
    validation_data = data[int(0.80 * datapoints):]
    data_mean, data_std = torch.mean(training_data), torch.std(training_data)
    # Modules Setup
    criterion = nn.BCELoss()
    error_props = BitErrorProb(power, alpha, distance)
    reception_props = PacketReceptionProb(packet_length)
    models = {'mean': (0.05, Network(layers, neurons)), 'max': (0.6, Network(layers, neurons))}
    # Training
    for reducing_method, (penalty_weight, network) in models.items():
        optimizer = optim.RMSprop(network.parameters(), lr=1e-4)
        metrics = list()
        network.train()
        for iteration in range(iterations):
            optimizer.zero_grad()
            # Selecting a Random Batch
            random_indices = np.random.randint(past_window * sample_rate, int(0.80 * datapoints) - future_window * sample_rate, batch_size)
            past_windows = torch.cat([training_data[index - past_window * sample_rate: index] for index in random_indices], dim=1)
            future_windows = torch.cat([training_data[index: index + future_window * sample_rate] for index in random_indices], dim=1)
            # Forward Pass through the Network Module
            past_windows_downsampled = func.interpolate(past_windows.permute(1, 2, 0), scale_factor=device_sample_rate / sample_rate, mode='linear').permute(2, 0, 1)
            past_windows_normalized = (past_windows_downsampled - data_mean) / data_std
            blacklist = network(past_windows_normalized)
            # Forward Pass through the Simulation Module
            interference_power_levels, channels_matrix = tsch(future_windows)
            error_prop_values = error_props(interference_power_levels)
            error_props_per_channel = torch.mul(error_prop_values.unsqueeze(dim=2), channels_matrix)
            errors_reduced = reduce(error_props_per_channel, reducing_method, channels_matrix)
            # Loss Function
            whitelist = torch.ones_like(blacklist) - blacklist
            outputs = torch.mul(errors_reduced, whitelist)
            desired_outputs = torch.zeros_like(outputs)
            cross_entropy_loss = criterion(outputs, desired_outputs)
            blacklisting_penalty = torch.mean(blacklist)
            loss_func = cross_entropy_loss + blacklisting_penalty * penalty_weight
            # Backward Pass
            loss_func.backward()
            # Optimization Step
            optimizer.step()
            # Logging
            iteration_metrics = (cross_entropy_loss.item(), blacklisting_penalty.item(), loss_func.item())
            metrics.append(iteration_metrics)
            print('Iteration {}/{} Cross-Entropy Loss: {:.4f} Blacklisting Penalty: {:.4f} Total Loss: {:.4f}'.format(
                iteration + 1, iterations, *iteration_metrics
            ))
        files_path = f'results/{dataset}-{reducing_method}'
        torch.save(network, files_path + '.pt')
        metrics = np.array(metrics)
        figure, axes = plt.subplots(3, 1, figsize=(5, 9), sharex=True)
        for dim, name in enumerate(('Cross-Entropy Loss', 'Penalty', 'Total Loss')):
            axes[dim].set_title(name)
            axes[dim].plot(metrics[:, dim])
        plt.tight_layout()
        plt.savefig(files_path + '-metrics.png')
        plt.show()
    # Evaluation
    figure, axes = plt.subplots(6, 1, sharex=True, sharey=True, figsize=(10, 15))
    with torch.no_grad():
        print(f'Dataset: "{dataset}.pt"')
        # Standard TSCH & Enhanced TSCH
        tsch_interference, tsch_channels_matrix = tsch(data)
        tsch_errors = error_props(tsch_interference)
        tsch_receptions = reception_props(tsch_errors)
        describe(tsch_receptions, 'TSCH')
        enhanced_tsch_interference = enhanced_tsch(data, device_sample_rate / sample_rate)
        enhanced_tsch_errors = error_props(enhanced_tsch_interference)
        enhanced_tsch_receptions = reception_props(enhanced_tsch_errors)
        describe(enhanced_tsch_receptions, 'Enhanced TSCH')
        for subplot in range(6):
            axes[subplot].plot(tsch_receptions.numpy().flatten(), label='Standard TSCH', color='blue')
            axes[subplot].plot(enhanced_tsch_receptions.numpy().flatten(), label='Enhanced TSCH', color='orange')
        # Intelligent TSCH
        for reducing_method, (penalty_weight, network) in models.items():
            print(f'Reducing Method: "{reducing_method}"')
            plt_color = 'green' if reducing_method == 'mean' else 'purple'
            network.eval()
            pivots = np.arange(past_window * sample_rate, datapoints, future_window * sample_rate)
            past_windows = torch.cat([data[index - past_window * sample_rate: index] for index in pivots], dim=1)
            past_windows_downsampled = func.interpolate(past_windows.permute(1, 2, 0), scale_factor=device_sample_rate / sample_rate, mode='linear').permute(2, 0, 1)
            past_windows_normalized = (past_windows_downsampled - data_mean) / data_std
            blacklist = network(past_windows_normalized)
            future_windows = torch.cat([data[index: index + future_window * sample_rate] for index in pivots], dim=1)
            subplot = 0
            for threshold in (0.25, 0.50, 0.75):
                available_channels = blacklist < threshold
                interference_values = intelligent_tsch(future_windows, available_channels)
                error_props_values = error_props(interference_values)
                reception_props_values = reception_props(torch.cat((tsch_errors[:past_window * sample_rate], error_props_values.permute(1, 0).view(-1, 1)), dim=0))
                describe(reception_props_values, 'Threshold: {:.2f}'.format(threshold))
                axes[subplot].plot(reception_props_values, label=reducing_method, color=plt_color)
                axes[subplot].set_title('Threshold: {:.2f}'.format(threshold))
                subplot += 1
            for n in (3, 6, 9):
                scores_sorted = np.sort(blacklist)
                thresholds = scores_sorted[:, n]
                thresholds = np.expand_dims(thresholds, axis=1)
                available_channels = blacklist < torch.from_numpy(thresholds)
                error_props_values = error_props(interference_values)
                reception_props_values = reception_props(torch.cat((tsch_errors[:past_window * sample_rate], error_props_values.permute(1, 0).view(-1, 1)), dim=0))
                describe(reception_props_values, 'Top {}'.format(n))
                axes[subplot].plot(reception_props_values, label=reducing_method, color=plt_color)
                axes[subplot].set_title('Top {}'.format(n))
                subplot += 1
    plt.savefig(f'results/{dataset}-performances.png')
    plt.show()
