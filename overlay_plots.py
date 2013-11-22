# AUTHOR: Sean Peters

from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import math
import time
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

top_dir = '/mnt/ict_test/jpeg2000/'
data_dir = ['4000', '7000', '10100']
param_dir = ['/qstep/', '/cblks/', '/precincts/', '/clevels/']
param_save = ['qstep', 'cblks', 'precincts', 'clevels']
param = ['Quantization step size', 'Block size', 'Precinct size', 'DWT levels']
param_log = [True, True, True, False]
dataset = ['A', 'B', 'C']

font = {'family' : 'sans-serif', 'style' : 'normal', 'size' : 14}

matplotlib.rc('font', **font)
mrk_sz = 50
lgnd_sz = 15

indices = [[],[],[]]
enc = [[],[],[]]
dec = [[],[],[]]
cr = [[],[],[]]
rmse = [[],[],[]]

for di, d in enumerate(data_dir):
  for pi, p in enumerate(param_dir):
    A = open(top_dir + data_dir[di] + param_dir[pi] + 'jp2_plots.txt', 'r').read()
    commas = 0
    for a in A:
      if a == ',':
        commas += 1
    length = commas / 5 + 1
    print (A)
    A = A.replace('\n', '').replace('[', '').replace(']', ',').replace(' ','').split(',')
    A = A[0:len(A)-1]
    print (A)
    A = list(map(float, A))

    indices[di].append (A[0:length])
    enc[di].append (A[length:length*2])
    dec[di].append (A[length*2:length*3])
    cr[di].append (A[length*3:length*4])
    rmse[di].append (A[length*4:length*5])


for pi, p in enumerate(param):
  plt.clf()
  plt.ylabel('Encoding time (s)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], enc[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('enc_' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  plt.ylabel('Decoding time (s)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], dec[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('dec_' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  plt.ylabel('Compression Ratio')
  if (param_save[pi] == 'qstep'):
    plt.yscale('log')

  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], cr[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  if (param_save[pi] == 'clevels'):
    plt.legend(loc=4,prop={'size':lgnd_sz})
  else:
    plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('cr_' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference ')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], rmse[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  #plt.ticklabel_format(style='sci', axis='y') 
  plt.grid()
  if (param_save[pi] == 'clevels'):
    plt.legend(loc=1,prop={'size':lgnd_sz})
  else:
    plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse_' + param_save[pi] + '.eps')
