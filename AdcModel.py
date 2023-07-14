import numpy as np

class ADC_MODEL:
    def __init__(self, fs=1e6, n_bits=16, full_scale=3.3):
        self.fs = fs
        self.n_bits = n_bits
        self.full_scale = full_scale
        
    def packg(self, signal, trigger):
        # Creates packages of data to be sampled. Eliminates the data
        # measured outside the sampling window defined by the trigger.
        packg = []
        for a in trigger:
            start_pack_index = np.where(signal.x >= a.start)[0][0]
            stop_pack_index  = np.where(signal.x > a.stop)[0][0]-1

            packg.append({'time': signal.x[start_pack_index:stop_pack_index], 
                          'amp': signal.y[start_pack_index:stop_pack_index]})
        return packg

    def sample(self, packg):
        # Sample data packages in time without discretizing the amplitude      
        ideal_sample = []
        for i, a in enumerate(packg):
            sampled_time = np.arange(a['time'][0], a['time'][-1], 1/self.fs)

            for b in sampled_time:
                ix = np.argmin(np.abs(packg[0][i]['time'] - b))
                ideal_sample.append({'time': a['time'][ix],
                                     'amp': a['amp'][ix]})
        return ideal_sample

    def discretize(self, ideal_sample):
        # Amplitude discretization around LSB/2     
        out = []
        for a in ideal_sample:
            bound_sample = np.maximum(np.minimum(a['amp'], self.full_scale), 0)  # bound values

            LSB = self.full_scale/(2**self.n_bits-1)  # amplitude discretization
            out.append({'amp': LSB*np.round(bound_sample/LSB), 'time': a['time']})
        return out

    def avg_pp(self, dig_signal, n_avg):
        # Averages n_avg peak-to-peak amplitudes
        out = []
        for a in dig_signal:
            out.append(np.sum(np.max(a['amp'], axis=0)[:n_avg] - np.min(a['amp'], axis=0)[:n_avg]) / n_avg)
        # making it compatible with inv_solve()
        out = np.transpose(out)
        return out

    def avg(self, dig_signal):
        # Averages all points (DC excitation)
        out = []
        for a in dig_signal:
            out.append(np.sum(a['amp'], axis=0) / len(a['amp']))
        # making it compatible with inv_solve()
        out = np.transpose(out)
        return out

