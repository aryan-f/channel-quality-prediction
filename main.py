import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from model import Network
from protocols import tsch, enhanced_tsch, intelligent_tsch
from simulation import BitErrorProb, PacketReceptionProb


def reduce(values, method, matrix=None):
    if method == 'mean':
        channels_sums = torch.sum(values, dim=0)
        channels_weights = torch.sum(matrix, dim=0)
        return channels_sums / channels_weights
    values, indices = torch.max(values, dim=0)
    return values


def describe(receptions, title, ed_enabled=True):
    print(title)
    prr = torch.mean(receptions[240 * sample_rate: 300 * sample_rate])
    n_ed, n_rx, n_tx = 3 if ed_enabled else 0, 7, 1
    i_ed, i_rx, i_tx = 5e-3, 5e-3, 10e-3
    t_ed, t_tx = 128e-6,  3.2e-3
    v_cc = 3.3
    energy = (i_ed * n_ed * t_ed + (i_rx * n_rx * t_tx + i_tx * n_tx * t_tx) / prr) * v_cc
    print('PRR: {:.4f}, Energy Consumption: {:.2f} uJ'.format(prr, energy * 1e6))


if __name__ == '__main__':
    # Adjusting the DPI of the figures.
    mpl.rcParams['figure.dpi'] = 300
    # Configuration
    dataset = 'apartments'
    sample_rate, device_sample_rate = 2000, 10
    power, alpha, distance = -10., 3.5, 3.
    packet_length = 133  # Bytes
    past_window, future_window = 5, 5  # Seconds
    layers, neurons = 2, 50
    iterations, batch_size = 1000, 32
    # Dataset Preparation
    data = torch.load(f'data/{dataset}.pt')
    datapoints, sequences, channels = data.shape
    training_data = data[:240 * sample_rate]
    validation_data = data[240 * sample_rate:]
    data_mean, data_std = torch.mean(training_data), torch.std(training_data)
    # Modules Setup
    criterion = nn.BCELoss()
    error_props = BitErrorProb(power, alpha, distance)
    reception_props = PacketReceptionProb(packet_length)
    models = {'mean': (0.05, Network(layers, neurons)), 'max': (0.55, Network(layers, neurons))}
    # Whether to train a new model or load pre-trained ones.
    user_input = input('Train again? [Y/n]')
    if user_input == 'Y':
        # Training a New Model
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
            # noinspection PyTypeChecker
            figure, axes = plt.subplots(3, 1, figsize=(5, 9), sharex=True)
            for dim, name in enumerate(('Cross-Entropy Loss', 'Penalty', 'Total Loss')):
                axes[dim].set_title(name)
                axes[dim].plot(metrics[:, dim])
            plt.tight_layout()
            plt.savefig(files_path + '-metrics.png')
            plt.show()
    else:
        # Loading already-trained Models
        for reducing_method, (penalty_weight, network) in models.items():
            model_path = f'results/{dataset}-{reducing_method}.pt'
            models[reducing_method] = (penalty_weight, torch.load(model_path))
    # Evaluation
    print(f'[{dataset.capitalize()}]')
    tsch_interference, tsch_channels_matrix = tsch(data)
    tsch_errors = error_props(tsch_interference)
    tsch_receptions = reception_props(tsch_errors)
    describe(tsch_receptions, 'Standard TSCH', ed_enabled=False)
    enhanced_tsch_interference = enhanced_tsch(data, device_sample_rate / sample_rate)
    enhanced_tsch_errors = error_props(enhanced_tsch_interference)
    enhanced_tsch_receptions = reception_props(enhanced_tsch_errors)
    describe(enhanced_tsch_receptions, 'Enhanced TSCH')
    # noinspection PyTypeChecker
    figure, axis = plt.subplots(sharex=True, sharey=True, figsize=(4, 2.5))
    axis.plot(tsch_receptions.numpy().flatten(), label='Standard TSCH', color='gray')
    axis.plot(enhanced_tsch_receptions.numpy().flatten(), label='Enhanced TSCH', color='orange')
    for reducing_method, (penalty_weight, network) in models.items():
        network.eval()
        pivots = np.arange(past_window * sample_rate, datapoints, future_window * sample_rate)
        past_windows = torch.cat([data[index - past_window * sample_rate: index] for index in pivots], dim=1)
        future_windows = torch.cat([data[index: index + future_window * sample_rate] for index in pivots], dim=1)
        past_windows_downsampled = func.interpolate(past_windows.permute(1, 2, 0), scale_factor=device_sample_rate / sample_rate, mode='linear').permute(2, 0, 1)
        past_windows_normalized = (past_windows_downsampled - data_mean) / data_std
        with torch.no_grad():
            blacklist = network(past_windows_normalized)
            scores_sorted = np.sort(blacklist)
            thresholds = scores_sorted[:, 8]
            thresholds = np.expand_dims(thresholds, axis=1)
            available_channels = blacklist < torch.from_numpy(thresholds)
            interference_values = intelligent_tsch(future_windows, available_channels)
            error_props_values = error_props(interference_values)
            reception_props_values = reception_props(torch.cat((tsch_errors[:past_window * sample_rate], error_props_values.permute(1, 0).view(-1, 1)), dim=0))
        describe(reception_props_values, 'Our Work ({})'.format(reducing_method))
        axis.plot(reception_props_values, label='Our Work ({})'.format(reducing_method), color='green' if reducing_method == 'mean' else 'blue')
        axis.set_xlabel('Time (Sec)')
        axis.set_ylabel('PRP')
        axis.set_xticks(np.arange(0, datapoints + 1, 15 * sample_rate))
        axis.set_xticklabels(np.arange(0, datapoints / sample_rate + 1, 15).astype('int'))
        axis.set_xlim(240 * sample_rate, min(len(reception_props_values), 300 * sample_rate))
        axis.set_ylim(0.7, 1.02)
        axis.set_title(dataset.capitalize())
    plt.show()
