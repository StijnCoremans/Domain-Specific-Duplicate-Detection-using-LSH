from typing import Dict, Set, List

import numpy as np
from numpy.typing import NDArray

def lsh(signature_matrix: NDArray, b: int, r: int, sep='-'):
    
    buckets_per_band = []
    n, products = signature_matrix.shape

    for band in range(b):
        buckets: Dict[str, Set[int]] = {}

        for j in range(products):
            current_band = signature_matrix[(band * r):((band + 1) * r), j]
            bucket_key = sep.join(map(str, current_band))
            buckets.setdefault(bucket_key, set()).add(j)

        buckets_per_band.append(buckets)

    same_bucket_counter = np.zeros((products, products))

    for band_buckets in buckets_per_band:
        for bucket in band_buckets.values():
            for i in bucket:
                for j in bucket:
                    if i == j:
                        continue
                    same_bucket_counter[min(i, j), max(i, j)] += 1

    same_bucket_counter /= 2
    candidates = np.argwhere(same_bucket_counter >= 1)
    return candidates
