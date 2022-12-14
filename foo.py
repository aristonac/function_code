import statistics
import math

xi = [
0.219727,
0.617188,
0.411133,
0.32959,
0.518555,
0.730957,
0.104492,
0.42334,
0.629395,
0.208008,
0.519043,
0.733887,
0.305664,
0.108887
]


yi = [
  0.228,
  0.229333,
  0.236,
  0.418,
  0.421333,
  0.432667,
  0.456,
  0.601333,
  0.614667,
  0.644667,
  0.798667,
  0.805333,
  0.825333,
  0.838
]

xo = [
0.126953,
0.319946,
0.519775,
0.221436,
0.618530,
0.418213,
0.009155,
0.325562,
0.513550,
0.727661,
0.104980,
0.420166,
0.626953,
0.205933,
0.010742,
0.008545,
0.522705,
0.729980,
0.301147,
0.107178
]

yo = [
0.065333,
0.071167,
0.071833,
0.227333,
0.233333,
0.241167,
0.274167,
0.418000,
0.421333,
0.431167,
0.454500,
0.604833,
0.612333,
0.645167,
0.646333,
0.650667,
0.798667,
0.802167,
0.825000,
0.835833
]

w = 4096
h = 3000
def euk_dist(x1, y1, x2, y2):
  x1 *= w
  x2 *= w
  y1 *= h
  y2 *= h
  return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

all_distances = []
distances = {}
for (i, x1) in enumerate(xi):
  y1 = yi[i]
  dist_min = w*h
  for (j, x2) in enumerate(xo):
    y2 = yo[j]
    dist = euk_dist(x1, y1, x2, y2)

    if i == 1:
      all_distances.append(dist)

    if dist < dist_min:
      distances[i] = dist
      dist_min = dist


print(distances)
print(statistics.mean(distances.values()))
print(statistics.median(distances.values()))