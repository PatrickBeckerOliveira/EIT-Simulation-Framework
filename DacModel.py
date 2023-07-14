import numpy as np

class DAC_MODEL:
    """
    Simulates digital-to-analog conversion. Can be used to sample an
    arbitrary signal vector or directly generate discretized
    pre-configured signals, such as sine waves.
    
    Used to generate PWL signals for PSPICE sources
    """

    def __init__(self, fs=1e6, n_bits=12, full_scale=3.3):
        self.fs = fs  # sampling frequency
        self.n_bits = n_bits  # resolution
        self.full_scale = full_scale  # full-scale voltage
    
    def sample(self, signal):
        """
        Samples arbitrary signal
        """
        out = []
        for a in signal:
            sampled_time = np.arange(a['time'][0], a['time'][-1], 1/self.fs)

            for b in sampled_time:
                ix = np.argmin(np.abs(a['time'] - b))
                out.append({'time': a['time'][ix], 'amp': a['amp'][ix]})
        return out

    def discretize(self, ideal_sample):
        """
        Discretizes sampled signal amplitude
        """
        out = []
        for a in ideal_sample:
            offset_sample = self.full_scale/2 + a['amp']
            offset_sample = np.maximum(np.minimum(offset_sample, self.full_scale), 0)  # bound values

            LSB = self.full_scale/(2**self.n_bits-1)  # amplitude discretization
            out.append({'amp': LSB*np.round(offset_sample/LSB), 'time': a['time']})
        return out

    def sine(self, f_sig, v_amp, n_periods):
        """
        Generates DAC sine wave
        """
        DA_time = np.arange(0, n_periods/f_sig, 1/self.fs)  # D/A time discretization
        ideal_sig = v_amp * np.sin(2 * np.pi * f_sig * DA_time)  # Signal definition
        offset_sig = self.full_scale/2 + ideal_sig
        offset_sig = np.maximum(np.minimum(offset_sig, self.full_scale), 0)  # bound values

        LSB = self.full_scale/(2**self.n_bits-1)  # D/A amplitude discretization
        out = {'amp': np.transpose(LSB * np.round(offset_sig/LSB)), 'time': np.transpose(DA_time)}
        return out

