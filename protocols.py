import math

import numpy as np
import torch
import torch.nn.functional as func


def tsch(measurements):
    # IEEE 802.15.4e Standard: Time-Slotted Channel Hopping
    datapoints, sequences, channels = measurements.shape
    used_channels = np.resize(np.arange(16).repeat(20), datapoints)
    channels_matrix = np.zeros(measurements.shape)
    channels_matrix[np.arange(datapoints), :, used_channels] = 1
    return measurements[np.arange(datapoints), :, used_channels], torch.from_numpy(channels_matrix)


def enhanced_tsch(measurements, scale_factor, alpha=0.1, selection_period=160, use_best_n=8):
    # Enhanced Time-Slotted Channel Hopping
    datapoints, sequences, channels = measurements.shape
    selection_period_length = math.ceil(selection_period / scale_factor)
    downsampled = func.interpolate(measurements.permute(1, 2, 0), scale_factor=scale_factor, mode='linear', align_corners=False).permute(2, 0, 1)
    downsampled = torch.squeeze(downsampled)
    downsampled = downsampled.numpy()
    qualities = downsampled
    for i in range(1, len(qualities)):
        qualities[i] = alpha * downsampled[i] + (1 - alpha) * qualities[i - 1]
    channels_used = np.resize(np.arange(16).repeat(20), selection_period_length)  # First period, when we don't have any CQE.
    pivots = np.arange(selection_period, len(downsampled), selection_period)
    for pivot in pivots:
        index = pivot - 1
        channels_qualities = qualities[index]
        best_channels = np.argsort(channels_qualities)[:use_best_n]
        channels_used_in_period = np.resize(best_channels.repeat(20), selection_period_length)
        channels_used = np.concatenate((channels_used, channels_used_in_period))
    channels_used = channels_used[:datapoints]  # The latest period might go further than the length of the measurements.
    return measurements[np.arange(datapoints), :, channels_used]


def intelligent_tsch(measurements, available):
    # Time-Slotted Channel Hopping with Intelligent Blacklisting
    datapoints, sequences, channels = measurements.shape
    assert available.shape == (sequences, channels)
    logged = False
    if not np.count_nonzero(available, axis=1).all():
        # Ensures that even in the worst circumstances, at least one channel is kept open.
        if not logged:
            print('No channel was available.')
            logged = True
        for row in available:
            if not row.any():
                row[-1] = True
    available = [np.resize(np.argwhere(li.numpy()).flatten().repeat(20), datapoints) for li in available]
    interferences = [measurements[np.arange(datapoints), i, available[i]] for i in range(sequences)]
    channels_matrix = np.zeros(measurements.shape)
    for i in range(sequences):
        channels_matrix[np.arange(datapoints), i, available[i]] = 1
    return torch.stack(interferences).permute(1, 0), channels_matrix
