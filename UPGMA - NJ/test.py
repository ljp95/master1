
"""
Created by ljp95 on 12.7.2018

"""

import UPGMA_NJ

''' Some examples '''
#Additivity
M1 = [[0,8,7,12], [8,0,9,14], [7,9,0,11], [12,14,11,0]]
#Ultrametric
M2 = [[0,2,3,8,14,18],[2,0,3,8,14,18],
      [3,3,0,8,14,18],[8,8,8,0,14,18],
      [14,14,14,14,0,18],[18,18,18,18,18,0]]
#UPGMA
M3 = [[0,19,27,8,33,18,13],[19,0,31,18,36,1,13],
      [27,31,0,26,41,32,29],[8,18,26,0,31,17,14],
      [33,36,41,31,0,35,28],[18,1,32,17,35,0,12],
      [13,13,29,14,28,12,0]]
#Neighbor Joining
M4 = [[0,2,4,6,6,8],[2,0,4,6,6,8],
      [4,4,0,6,6,8],[6,6,6,0,4,8],
      [6,6,6,4,0,8],[8,8,8,8,8,0]]

print("M1 : {}".format(M1))
print("M2 : {}".format(M2))
print("M3 : {}".format(M3))
print("M4 : {}".format(M4))

print("M1 additive ? {}".format(verifyAdditivity(M1)))
print("M1 ultrametric ? {}".format(verifyUltrametric(M1)))
print("M2 ultrametric ? {}".format(verifyUltrametric(M2)))
print("UPGMA : {}".format(upgma(M3)))
print("Neighbour Joining : {}".format(NeighbourJoining(M4)))



