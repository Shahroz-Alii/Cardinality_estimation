import mmh3
import math

class HyperLogLog:
    def __init__(self, b):
        self.b = b
        self.m = 2**b
        self.registers = [0] * self.m
        # Alpha constants for HLL bias correction
        if self.m == 16: self.alpha = 0.673
        elif self.m == 32: self.alpha = 0.697
        elif self.m == 64: self.alpha = 0.709
        else: self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def add(self, item):
        h = mmh3.hash(str(item), signed=False) & 0xFFFFFFFF
        idx = h >> (32 - self.b)
        w = h & ((1 << (32 - self.b)) - 1)
        
        if w == 0:
            rho = (32 - self.b) + 1
        else:
            rho = (32 - self.b) - w.bit_length() + 1
            
        self.registers[idx] = max(self.registers[idx], rho)

    def estimate(self):
        Z = sum(2.0**-r for r in self.registers)
        E = self.alpha * (self.m**2) / Z
        
        # Small range correction (Linear Counting)
        if E <= 2.5 * self.m:
            V = self.registers.count(0)
            if V > 0:
                E = self.m * math.log(self.m / V)
        return E

class Recordinality:
    def __init__(self, k):
        self.k = k
        self.S = set()
        self.modifications = 0

    def add(self, item):
        h = mmh3.hash(str(item), signed=False) & 0xFFFFFFFF
        
        if len(self.S) < self.k:
            if h not in self.S:
                self.S.add(h)
                self.modifications += 1
        else:
            # We track 'k' largest values
            min_val = min(self.S)
            if h > min_val and h not in self.S:
                self.S.remove(min_val)
                self.S.add(h)
                self.modifications += 1

    def estimate(self):
        m = self.modifications
        if m <= self.k: return float(len(self.S))
        return self.k * math.pow(1 + 1/self.k, m - self.k) - 1
    

class KMV:
    def __init__(self, k):
        self.k = k
        self.min_values = set()

    def add(self, item):
        # Hash to a float in [0, 1]
        h = mmh3.hash(str(item), seed=42, signed=False) / 0xFFFFFFFF
        if len(self.min_values) < self.k:
            self.min_values.add(h)
        elif h < max(self.min_values) and h not in self.min_values:
            self.min_values.remove(max(self.min_values))
            self.min_values.add(h)

    def estimate(self):
        if len(self.min_values) < self.k:
            return float(len(self.min_values))
        u_k = max(self.min_values)
        return (self.k - 1) / u_k