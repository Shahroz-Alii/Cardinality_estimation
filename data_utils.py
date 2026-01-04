import numpy as np
import re

def generate_zipf_stream(n, N, alpha):
    """
    Routines to generate synthetic data streams Z as per prompt.
    n: number of distinct elements
    N: total length of the stream
    alpha: Zipfian parameter (skewness)
    """
    # 1. Calculate ranks 1 to n
    ranks = np.arange(1, n + 1)
    
    # 2. Calculate probabilities: 1 / i^alpha
    weights = 1.0 / np.power(ranks, alpha)
    
    # 3. Normalize with the constant cn (weights.sum() is 1/cn)
    weights /= weights.sum() 
    
    # 4. Generate the stream by picking 'ranks' based on Zipfian weights
    stream = np.random.choice(ranks, size=N, p=weights)
    return stream

def get_words_from_file(filename):
    """Cleans text as per assignment: lowercase, 3+ chars."""
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        words = re.findall(r'\b[a-z]{3,}\b', text)
    return words