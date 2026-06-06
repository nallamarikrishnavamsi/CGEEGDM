import pickle
from tqdm import tqdm
from glob import glob
import numpy as np
from scipy.stats import describe
from src.util import staged_mu_law


class WelfordVariance:
    def __init__(self):   # Comparison to ShiftDataVariance:
        self.mean = 0.0   # = K + Ex / n
        self.count = 0    # = n
        self.M2 = 0.0     # = Ex2 - (Ex)^2 / n

    def add_variable(self, x: float):
        self.count += 1
        old_mean = self.mean
        self.mean += (x - self.mean) / self.count
        self.M2 += (x - old_mean) * (x - self.mean)

    def remove_variable(self, x: float):
        self.count -= 1
        new_mean = self.mean
        self.mean -= (x - self.mean) / self.count
        self.M2 -= (x - new_mean) * (x - self.mean)
        
    def get_mean(self) -> float:
        return self.mean

    def get_variance(self) -> float:
        return self.M2 / self.count
    
    def get_sample_variance(self) -> float:
        return self.M2 / (self.count - 1)

def merge(a: WelfordVariance, b: WelfordVariance) -> WelfordVariance:
    ab = WelfordVariance()
    ab.count = a.count + b.count
    delta = b.mean - a.mean
    ab.mean = (a.count * a.mean + b.count * b.mean) / ab.count
    ab.M2 = a.M2 + b.M2 + delta**2 * a.count * b.count / ab.count
    return ab
    

all_dir = ["train", "val", "test"]

for dir in all_dir:
    n_out_neg = 0
    n_out_pos = 0
    n_total = 0
    data_min = float("inf")
    data_max = float("-inf")

    data_min_mu = float("inf")
    data_max_mu = float("-inf")

    welford = None
    welford_mu = None

    all_data_path = glob(f"data/faithful/{dir}/*.pkl")
    for data_path in tqdm(all_data_path):
        with open(data_path, "rb") as f:
            signal = pickle.load(f)["signal"]
            n_out_neg += (signal < -1).astype(int).sum()
            n_out_pos += (signal > 1).astype(int).sum()
            n_total += np.prod(signal.shape)
            data_min = min(signal.min(), data_min)
            data_max = max(signal.max(), data_max)

            new_welford = WelfordVariance()
            new_welford.count = np.prod(signal.shape)
            new_welford.mean = signal.mean()
            new_welford.M2 = ((signal - signal.mean()) ** 2).sum()
            
            if welford is None: welford = new_welford
            else: welford = merge(welford, new_welford)


            signal = staged_mu_law(signal)
            data_min_mu = min(signal.min(), data_min_mu)
            data_max_mu = max(signal.max(), data_max_mu)
            
            new_welford = WelfordVariance()
            new_welford.count = np.prod(signal.shape)
            new_welford.mean = signal.mean()
            new_welford.M2 = ((signal - signal.mean()) ** 2).sum()
            
            if welford_mu is None: welford_mu = new_welford
            else: welford_mu = merge(welford_mu, new_welford)

    print(dir, n_out_neg, n_out_pos, n_total, n_out_neg / n_total, n_out_pos / n_total)
    print(data_min, data_max, welford.get_variance() ** 0.5, welford.get_mean())
    print(data_min_mu, data_max_mu, welford_mu.get_variance() ** 0.5, welford_mu.get_mean())