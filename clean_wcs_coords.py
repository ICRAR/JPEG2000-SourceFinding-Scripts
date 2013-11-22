from __future__ import print_function
from copy import deepcopy
import sys
import numpy as np
import os
import math
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
wcs_coords = open('wcs_coords.txt', 'r').read().split()
xyz_coords = open('xyz_coords.txt', 'r').read().split()

j = 0
with open('new_wcs_coords.txt', "w+") as wcs:
  for i in xrange(len(xyz_coords)/7):
    while (not xyz_coords[i*7] == wcs_coords[j*4]):
      j+=1
    print (str(wcs_coords[j*4]) + ' ' + 
           str(wcs_coords[j*4+1]) + ' ' +
           str(wcs_coords[j*4+2]) + ' ' +
           str(wcs_coords[j*4+3]), file=wcs)
