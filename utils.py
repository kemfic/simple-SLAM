import numpy as np

def hamming_distance(a, b):
  '''
  Computes Hamming distance between two vectors.
  https://stackoverflow.com/q/40875282
  '''
  r = (1 << np.arange(8))[:,None]
  return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)
