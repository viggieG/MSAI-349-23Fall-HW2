import numpy as np

def euclid_dist(p1, p2)
  temp = p1 - p2
  euclid_dist = np.sqrt(np.dot(temp.T, temp))
  return euclid_dist
