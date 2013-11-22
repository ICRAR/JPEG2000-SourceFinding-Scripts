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
import decimal
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

class jp2_parameter_explorer():
  # paramters:
  #   RA - extent of RA dimension
  #   DEC - extent of DEC dimension
  #   FREQ - extent of FREQ dimension
  #   results_dir - dir to store the results
  #   skuareview_dir - location of skuareview encoder and decoder
  #   input_cube - FITS cube to be tested

  def __init__(self, RA, DEC, FREQ, input_cube,
      duchamp_comparer):
    self.RA = RA
    self.DEC = DEC
    self.FREQ = FREQ
    self.input_cube = input_cube
    self.fits_merge = '/home/speters/work/thesis/code/scripts/fits_tools/fits_merge'
    self.duchamp_comparer = duchamp_comparer

  def run_test(self, results_dir, jp2_param_name, encoder, decoder, domain,
      encoded_cube_name):

    # plot style init
    font = {'family' : 'sans-serif', 'style' : 'normal', 'size' : 14}
    matplotlib.rc('font', **font)
    markers_array = ['^', '^', 'v', 'v', '+', '+', 'x', 'x', '^', '^', 'v', 'v']
    colors_array = ['#AAAAAA', '#000000', '#AAAAAA',
    '#000000', '#AAAAAA', '#000000', '#AAAAAA', '#000000']
    mrk_sz = 50 #have to change in rmse_plot as well
    lgnd_sz = 15

    # the cube duchamp will run on
    encoded_cube_name = results_dir + '/' + encoded_cube_name
    output_cube = results_dir + '/outcube.fits'
    print ('\n################################')
    print ('Running test: ' + jp2_param_name)
    print ('################################')

    # make this test's directory
    with open(os.devnull, 'w') as fnull: 
      subprocess.call(['mkdir', results_dir], stdout = fnull, stderr = fnull)
      # directory to store temporary frames
      subprocess.call(['mkdir', results_dir + '/jpx_frames/'], stderr = fnull, 
          stdout = fnull)
      subprocess.call(['mkdir', results_dir + '/fits_frames/'], stderr = fnull, 
          stdout = fnull)
      # directory for duchamp results
      subprocess.call(['mkdir', results_dir + '/duchamp/'], stderr = fnull, 
          stdout = fnull)

    begin = domain[0] 
    end = domain[1]
    step_size = domain[2]
    indices = []
    
    # RMSE and compression ratio results arrays (for each iteration)
    # graphs calculated here
    enc_time = []
    dec_time = []
    rmse = [] # sample rmse
    cr = [] # compression ratio
    i = begin
      
    # graphs calculated in duchamp comparator 
    #0
    completeness = []
    soundness = []
    #1
    intflux = []
    w_vel = []
    incl = []
    hdiam = []
    #2
    ra_rmse = []
    dec_rmse = []
    freq_rmse = []
    wfreq_rmse = []
    intflux_rmse = []
    wra_rmse = []
    wdec_rmse = []

    p = subprocess.Popen(['stat', '-c', '%s', self.input_cube], \
        stdout=subprocess.PIPE)
    orig_size = int(p.communicate()[0])
    
    # For each step (i):
    # compress image, record samples A
    # decompress image, record samples B
    # calculate RMSE, Copmression Ratio and Source Finding results
    counter = 0
    i = begin
    while i <= end:
      print ('Testing:') 
      print (encoder(i, self.input_cube, encoded_cube_name, 
            self.RA, self.DEC, self.FREQ))
      print (decoder(encoded_cube_name, output_cube,
            self.RA, self.DEC, self.FREQ))
      indices.append (i)
      # Convert the control image from FITS to JPX (for each frame)
      # using jp2_param
      start = time.time()
      print ('Encoding cube\r')
      with open(os.devnull, 'w') as fnull: 
        subprocess.call(
            encoder(i, self.input_cube, encoded_cube_name, 
              self.RA, self.DEC, self.FREQ), 
            stdout = fnull)
      enc_time.append (time.time() - start)
      print ('Encoding time: ' + str(enc_time[-1]))

      # Calculate Compression Ratio after conversion
      p = subprocess.Popen(['stat', '-c', '%s', 
          encoded_cube_name], 
          stdout=subprocess.PIPE)
      cr.append (1.0 * orig_size / int(p.communicate()[0]))

      # Convert from JPX back to FITS
      start = time.time()
      print ('Decoding cube\r')
      with open(os.devnull, 'w') as fnull: 
        subprocess.call(
            decoder(encoded_cube_name, output_cube, 
              self.RA, self.DEC, self.FREQ),
            stdout = fnull)
      dec_time.append (time.time() - start)
      print ('Decoding time: ' + str(dec_time[-1]))

      # Calculate RMSE of origional FITS vs new FITS
      p = subprocess.Popen([
          '/home/speters/work/thesis/code/scripts/fits_tools/fits_rmse',
          self.input_cube, output_cube, self.RA, self.DEC, self.FREQ], \
          stdout=subprocess.PIPE)
      rmse.append (float(p.communicate()[0]))

      duc_results = self.duchamp_comparer.run_comparison(output_cube, 
          results_dir + '/duchamp/' + str(i) + '/', False)

      completeness.append(duc_results[0])
      soundness.append(duc_results[1])

      intflux.append(duc_results[2][0])
      w_vel.append(duc_results[2][1])
      incl.append(duc_results[2][2])
      hdiam.append(duc_results[2][3])

      ra_rmse.append(duc_results[3][0])
      dec_rmse.append(duc_results[3][1])
      freq_rmse.append(duc_results[3][2])
      intflux_rmse.append(duc_results[3][3])
      wfreq_rmse.append(duc_results[3][4])
      wra_rmse.append(duc_results[3][5])
      wdec_rmse.append(duc_results[3][6])

      # iterate cblk parameter
      if (domain[3] == '*'):
        i *= step_size
      else:
        i += step_size
      counter += 1

    with open(results_dir + '/jp2_plots.txt', "w+") as posi:
      print(str(indices), file = posi)
      print(str(enc_time), file = posi)
      print(str(dec_time), file = posi)
      print(str(cr), file = posi)
      print(str(rmse), file = posi)

    control_results = self.duchamp_comparer.get_control_graphs()

    # Finally create plots using python
    plt.clf()
    if (domain[4] == 'log'):
      plt.xscale('log')
    plt.scatter(indices, enc_time, marker = 'x')
    plt.ylabel('Encoding time (s)')
    plt.xlabel(jp2_param_name)
    plt.grid()
    plt.savefig(results_dir + '/' + 'encoding_time.eps')

    plt.clf()
    if (domain[4] == 'log'):
      plt.xscale('log')
    plt.scatter(indices, dec_time, marker = 'x')
    plt.ylabel('Decoding time (s)')
    plt.xlabel(jp2_param_name)
    plt.grid()
    plt.savefig(results_dir + '/' + 'decoding_time.eps')

    cr = [5.0345822484958989, 5.8612415665964335, 7.0079764706022942, 8.6648448595323284, 11.18110586377091, 15.012164276979906, 21.843825222814161, 37.069551471333227, 70.794591968071217, 150.5220586412012, 475.02499561078616, 2378.4859458812161, 16309.281183932348]

    plt.clf()
    plt.plot(cr, completeness, color = 'r', label = 'JPEG2000')
    plt.axhline(control_results[0], color = 'b', label = 'Uncompressed')
    plt.axhline(0.0892531876138, color = 'g', label = 'Uncompressed (a trous)')
    plt.xlabel('Compression Ratio')
    plt.xscale('log')
    plt.ylabel('Completeness')
    plt.grid()
    plt.legend(loc=3,prop={'size':10})
    plt.savefig(results_dir + '/' + 'seminar_completeness.eps')
    plt.savefig(results_dir + '/' + 'seminar_completeness.png')

    plt.clf()
    plt.plot(cr, soundness, color = 'r', label = 'JPEG2000')
    plt.axhline(control_results[1], color = 'b', label = 'Uncompressed')
    plt.axhline(0.590361445783, color = 'g', label = 'Uncompressed (a trous)')
    plt.xlabel('Compression Ratio')
    plt.xscale('log')
    plt.ylabel('Soundness')
    plt.grid()
    plt.legend(loc=3,prop={'size':10})
    plt.savefig(results_dir + '/' + 'seminar_soundness.eps')
    plt.savefig(results_dir + '/' + 'seminar_soundness.png')

    for indi, ind in enumerate(indices):                                          
      indices[indi] = self.round_sigfigs(ind, 3)                                  

    #
    # The A'trous axis is hard coded here, be sure to comment out the correct
    # line for the correct dataset (A, B or C)
    #

    plt.clf()                                                                     
    if (domain[4] == 'log'):                                                      
      plt.xscale('log')                                                           
    #plt.axhline(0.202749140893 - control_results[0], color = 'r', label = 'Atrous')                
    #plt.axhline(0.0892531876138 - control_results[0], color = 'r', label = 'Atrous')               
    #plt.axhline(0.029992684711 - control_results[0], color = 'r', label = 'Atrous')    
    plt.scatter(indices, self.diff_up(completeness, control_results[0]), marker   
        = 'x', s = mrk_sz, label = 'JPEG2000')                                                        
    plt.ylabel('Completeness Difference')                                         
    plt.xlim([indices[0],indices[-1]])                                            
    plt.xlabel(jp2_param_name)                                                    
    plt.grid()                                                                    
    plt.legend(loc=8,prop={'size':10})
    plt.savefig(results_dir + '/completeness.eps')                                

    plt.clf()                                                                     
    if (domain[4] == 'log'):                                                      
      plt.xscale('log')                                                           
    #plt.axhline(0.766233766234 - control_results[1], color = 'r', label = 'Atrous')                
    #plt.axhline(0.590361445783 - control_results[1], color = 'r', label = 'Atrous')                
    #plt.axhline(0.650793650794 - control_results[1], color = 'r', label = 'Atrous')
    plt.scatter(indices, self.diff_up(soundness, control_results[1]), marker =    
        'x', s = mrk_sz, label = 'JPEG2000')                                                          
    plt.ylabel('Soundness Difference')                                            
    plt.xlim([indices[0],indices[-1]])                                            
    plt.xlabel(jp2_param_name)                                                    
    plt.grid()                                                                    
    plt.legend(loc=8,prop={'size':10})
    plt.savefig(results_dir + '/soundness.eps')

    plt.clf()
    if (domain[4] == 'log'):
      plt.xscale('log')
      for r in rmse:
        r = r / 1E-6
    plt.yscale('log')
    plt.scatter(indices, rmse, marker = 'x')
    plt.ylabel('RMSE ($\times10^{-6})$')
    plt.xlabel(jp2_param_name)
    plt.grid()
    plt.savefig(results_dir + '/' + 'RMSE.eps')

    for fi, f in enumerate(freq_rmse): #convert from Hz to KHz
      freq_rmse[fi] = f / 1000
    for wfi, wf in enumerate(wfreq_rmse):
      wfreq_rmse[wfi] = wf / 1000
    

    # RMSE plots
    with open(results_dir + '/jp2_rmse_plots.txt', "w+") as posi:
      print(str(indices), file = posi)
      self.rmse_plot ('RMSE of Right Ascension (Deg)', jp2_param_name,
          control_results[3][0], ra_rmse, indices, results_dir + '/rmse-ra.eps',
          domain[4] == 'log', posi)
      self.rmse_plot ('RMSE of Declination (Deg)', jp2_param_name,
          control_results[3][1], dec_rmse, indices, results_dir + '/rmse-dec.eps',
          domain[4] == 'log', posi)
      self.rmse_plot ('RMSE of Frequency (KHz)', jp2_param_name,
          control_results[3][2], freq_rmse, indices, results_dir +
          '/rmse-freq.eps', domain[4] == 'log', posi)
      self.rmse_plot ('RMSE of Integrated Flux (mJy km/s)', 'Compression Ratio',
          control_results[3][3], intflux_rmse, cr, results_dir +
          '/rmse-intflux', domain[4] == 'log', posi)
      self.rmse_plot ('RMSE of Width Frequency (KHz)', jp2_param_name,
          control_results[3][4], wfreq_rmse, indices, results_dir +
          '/rmse-wfreq.eps', domain[4] == 'log', posi)
      self.rmse_plot ('RMSE of Right Ascension Width (Deg)', jp2_param_name,
          control_results[3][5], wra_rmse, indices, results_dir +
          '/rmse-wra.eps', domain[4] == 'log', posi)
      self.rmse_plot ('RMSE of Declination Width (Deg)', jp2_param_name,
          control_results[3][6], wdec_rmse, indices, results_dir +
          '/rmse-wdec.eps', domain[4] == 'log', posi)

    # Percentage identified wrt. some parameter, plots
    plt.clf()
    plt.xlabel('Integrated Flux (mJy km/s)')
    plt.ylabel('Percentage Identified')
    colors = iter(colors_array)
    markers = iter(markers_array)
    for zi, zz in enumerate(control_results[2][0][1]):
      control_results[2][0][1][zi] = zz * 100
    plt.scatter(control_results[2][0][0], control_results[2][0][1],
      label="control", marker=next(markers), color=next(colors), s = mrk_sz)
    for m in range(0, len(intflux), 2):
      for zi, zz in enumerate(intflux[m][1]):
        intflux[m][1][zi] = zz * 100
      plt.scatter(intflux[m][0], intflux[m][1], label=str(indices[m]), marker
          =next(markers), color=next(colors), s = mrk_sz)
    plt.legend(loc=4,prop={'size':10},title='quant. step size')
    plt.grid()
    plt.savefig(results_dir + '/intflux_perc.eps')

    plt.clf()
    ax = plt.gca()
    ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='both')
    plt.xlabel('Frequency Width (Hz)')
    plt.ylabel('Percentage Identified')
    colors = iter(colors_array)
    for zi, zz in enumerate(control_results[2][1][1]):
      control_results[2][1][1][zi] = zz * 100
    plt.scatter(control_results[2][1][0], control_results[2][1][1],
        label="control", marker = 'x', color=next(colors), s = mrk_sz)
    for m in range(0, len(w_vel), 2):
      for zi, zz in enumerate(w_vel[m][1]):
        w_vel[m][1][zi] = zz * 100
      plt.scatter(w_vel[m][0], w_vel[m][1], label=str(indices[m]), marker = 'x',
          color=next(colors), s = mrk_sz)
    plt.legend(loc=2,prop={'size':10})
    plt.grid()
    plt.savefig(results_dir + '/wfreq_perc.eps')

    plt.clf()
    plt.xlabel('Inclination (Rad)')
    plt.ylabel('Percentage Identified')
    colors = iter(colors_array)
    for zi, zz in enumerate(control_results[2][2][1]):
      control_results[2][2][1][zi] = zz * 100
    plt.scatter(control_results[2][2][0], control_results[2][2][1],
        label="control", marker = 'x', color=next(colors), s = mrk_sz)
    for m in range(0, len(incl), 2):
      for zi, zz in enumerate(incl[m][1]):
        incl[m][1][zi] = zz * 100
      plt.scatter(incl[m][0], incl[m][1], label=str(indices[m]), marker = 'x',
          color=next(colors), s = mrk_sz)
    plt.legend(loc=4,prop={'size':10})
    plt.grid()
    plt.savefig(results_dir + '/incl_perc.eps')

    plt.clf()
    plt.xlabel('HI Diameter (kpc)')
    plt.ylabel('Percentage Identified')
    colors = iter(colors_array)
    for zi, zz in enumerate(control_results[2][3][1]):
      control_results[2][3][1][zi] = zz * 100
    plt.scatter(control_results[2][3][0], control_results[2][3][1],
        label="control", marker = 'x', color=next(colors), s = mrk_sz)
    for m in range(0, len(hdiam), 2):
      for zi, zz in enumerate(hdiam[m][1]):
        hdiam[m][1][zi] = zz * 100
      plt.scatter(hdiam[m][0], hdiam[m][1], label=str(indices[m]), marker = 'x',
          color=next(colors), s = mrk_sz)
    plt.legend(loc=2,prop={'size':10})
    plt.grid()
    plt.savefig(results_dir + '/hdiam_perc.eps')

  def rmse_plot (self, ylab, xlab, control, rmse, indices, save, log, posi):
    print(str(self.diff_down(control,rmse)), file = posi)
    plt.clf()
    ax = plt.gca()
    ax.ticklabel_format(style='sci',scilimits=(-3,3),axis='y')

    if (log):
      plt.xscale('log')
    plt.plot(indices, rmse, color = 'r', label = 'JPEG2000')
    plt.axhline(control, color = 'b', label = 'Uncompressed')
    plt.xlim([0.0,indices[-1]])
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.grid()
    #s = ScalarFormatter()
    #ax.yaxis.set_major_formatter(s)
    plt.legend(loc=2,prop={'size':10})
    plt.savefig(save + ".png")
    plt.savefig(save + ".eps")

  def percent_ident (self, xlab, ylab, D, indices, control, save, log):
    plt.clf()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel('Percentage Identified')
    
    for m in xrange(len(D)):
      ax.scatter(xs=D[m][0], ys=[indices[m]]*len(D[m][0]), zs=D[m][1])
    fig.savefig(save + '.eps')

    plt.clf()
    plt.xlabel(xlab)
    plt.ylabel('Percentage Identified')
    plt.scatter(control[0], control[1], label="control")
    plt.grid()
    plt.savefig(save + '_control.eps')

  def pair_up (self, A, B):
    C = []
    for i in xrange(len(A)):
      C.append ([A[i], B[i]])
    return C

  def diff_up (self, A, B):
    C = []
    for i in xrange(len(A)):
      C.append (A[i] - B)
    return C

  def diff_down (self, A, B):
    C = []
    for i in xrange(len(B)):
      C.append (A - B[i])
    return C

  def round_sigfigs(self, num, sig_figs):
    if num != 0:
      return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
      return 0  # Can't take the log of 0
