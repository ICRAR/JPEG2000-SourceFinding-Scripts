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
rmse_ra = [[],[],[]]
rmse_dec = [[],[],[]]
rmse_freq = [[],[],[]]
rmse_intflux = [[],[],[]]
rmse_wra = [[],[],[]]
rmse_wdec = [[],[],[]]
rmse_wfreq = [[],[],[]]

for di, d in enumerate(data_dir):
  for pi, p in enumerate(param_dir):
    A = open(top_dir + data_dir[di] + param_dir[pi] + 'jp2_rmse_plots.txt', 'r').read()
    commas = 0
    for a in A:
      if a == ',':
        commas += 1
    length = commas / 8 + 1
    A = A.replace('\n', '').replace('[', '').replace(']', ',').replace(' ','').split(',')
    A = A[0:len(A)-1]
    A = list(map(float, A))

    indices[di].append (A[0:length])
    rmse_ra[di].append (A[length:length*2])
    rmse_dec[di].append (A[length*2:length*3])
    rmse_freq[di].append (A[length*3:length*4])
    rmse_intflux[di].append (A[length*4:length*5])
    rmse_wfreq[di].append (A[length*5:length*6])
    rmse_wra[di].append (A[length*6:length*7])
    rmse_wdec[di].append (A[length*7:length*8])

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference of RA (deg)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    print (indices[0][pi])
    print (rmse_ra[di][pi])
    plt.scatter(indices[0][pi], rmse_ra[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse-ra-' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference of Dec (deg)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], rmse_dec[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse-dec-' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference of Frequency (Hz)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], rmse_freq[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse-freq-' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference of RA width (deg)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], rmse_wra[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse-wra-' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference of Dec width (deg)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], rmse_wdec[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse-wdec-' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference of Freq width (Hz)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], rmse_wfreq[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse-wfreq-' + param_save[pi] + '.eps')

for pi, p in enumerate(param):
  plt.clf()
  ax = plt.gca()
  ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')
  plt.ylabel('RMSE Difference of Integrated Flux (mJy km/s)')
  plt.xlabel(param[pi])
  if (param_log[pi] == True):
    plt.xscale('log')
  plt.xlim(indices[0][pi][0], indices[0][pi][-1])
  colors = iter(['#000000','#AAAAAA','#000000'])
  markers = iter(['x','x','+'])
  for di, d in enumerate(data_dir): 
    plt.scatter(indices[0][pi], rmse_intflux[di][pi], marker = next(markers), color=next(colors),
        label=dataset[di], s = mrk_sz)
  plt.grid()
  plt.legend(loc=2,prop={'size':lgnd_sz})
  plt.savefig('rmse-intflux-' + param_save[pi] + '.eps')
