from typing import List, Set
import random
import numpy as np
from numpy.typing import NDArray

def minhash(binary_vectors: List[Set[int]], n: int, vector_length: int) -> NDArray:
    
    prime = 864158203
    if prime < vector_length:
        raise ValueError("Prime number p must be greater than the vocabulary size.")

    signature_matrix = np.full((n, len(binary_vectors)), np.inf)

    a_coef = np.array([random.randint(2**31, 2**32 - 1) for _ in range(n)])
    b_coef = np.array([random.randint(2**31, 2**32 - 1) for _ in range(n)])
   

    hash_values = np.empty((vector_length, n), dtype=int)
    for v in range(vector_length):
        hash_values[v] = (a_coef * v + b_coef) % prime
    
    hash_values_t = hash_values.T

    for product, indices in enumerate(binary_vectors):
        for non_zero in indices:
            signature_matrix[:, product] = np.minimum(signature_matrix[:, product], hash_values_t[:, non_zero])

    return signature_matrix.astype(int)