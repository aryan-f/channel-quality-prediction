import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

from protocols import tsch


class BitErrorProb(nn.Module):

    def __init__(self, trans_power, alpha, distance):
        super().__init__()
        self.path_loss = alpha * (20.1 + 10 * np.log10(distance))
        self.trans_power = torch.tensor(trans_power)
        self.fb_B = 10 * np.log10(2.5e5 / 2e6)

    def forward(self, interference):
        datapoints, sequences = interference.shape
        eb_n0 = self.trans_power - self.path_loss - interference - self.fb_B
        errors = torch.erfc(torch.sqrt(func.relu(eb_n0)))
        return torch.tensor(1 / 2) * errors


class PacketReceptionProb(nn.Module):

    def __init__(self, packet_length=133):
        super().__init__()
        self.packet_length = packet_length

    def forward(self, error_props):
        datapoints, sequences = error_props.shape
        receptions = torch.ones_like(error_props)
        success_props = 1 - error_props
        for k in range(8 * self.packet_length):
            offset = int(4 * k / 500)
            padded = torch.cat((success_props[offset:, :], torch.ones(offset, sequences)), dim=0)
            receptions = receptions.mul(padded)
        return torch.mean(receptions.unfold(dimension=0, size=2000, step=1), dim=2)


if __name__ == '__main__':
    # Let's see if the implemented modules result in the same figures as Tavakoli et al.
    # They also apply an averaging filter (with a window size of 2000 datapoints) which is not applied here.
    for filename in ('apartments', 'downtown', 'suburb'):
        measurements = torch.load(f'data/{filename}.pt')
        datapoints, sequences, channels = measurements.shape
        plt.figure(figsize=(10, 3))
        for trans_power in (4, 0, -10):
            for alpha in (2.5, 3.5):
                bit_error = BitErrorProb(trans_power, alpha, distance=3.0)
                packet_reception = PacketReceptionProb(packet_length=133)
                with torch.no_grad():
                    interference_values, channels_matrix = tsch(measurements)
                    error_props = bit_error(interference_values)
                    reception_props = packet_reception(error_props)
                plt.plot(reception_props.numpy().flatten(), label=fr'$P_{{tx}}={trans_power}, \alpha={alpha}$')
        plt.xticks(np.arange(0, datapoints + 1, 25 * 2000), np.arange(0, datapoints / 2000 + 1, 25, dtype='int'))
        plt.yticks(np.arange(0, 1.1, 0.2), np.arange(0, 101, 20))
        plt.title(filename.capitalize())
        plt.xlabel('Time [sec.]')
        plt.ylabel('PRP %')
        plt.xlim(0, datapoints)
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.show()
