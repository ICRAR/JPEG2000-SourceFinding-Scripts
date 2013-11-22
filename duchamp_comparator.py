from __future__ import print_function
from copy import deepcopy
import sys
import numpy as np
import os
import math
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DUCHAMP = '/home/speters/Duchamp-1.3.2/Duchamp-1.3.2'
# assuming beam size is 3 or 4 pixels
EPS = 10E-9

class duchamp_comparator():

  def __init__(self, trueset_file, control_image, control_results_dir, CRVAL3, FREQ):
    print ("Initialising 'duchamp_comparison'!")

    self.control_match = []
    # Read true source file list into memory
    words = open(trueset_file, 'r').read().split()
    wcs_coords = open('wcs_coords.txt', 'r').read().split()
    xyz_coords = open('xyz_coords.txt', 'r').read().split()

    #CRVAL3 is the starting frequency, this is listed in the FITS header,
    #at the moment I manually read this for each FITS file run this on
    print (CRVAL3)
    print (CRVAL3 + FREQ * -18518)
    self.true_set = []
    for i in xrange(len(words)/10):
      if (float(wcs_coords[i*4+3])*1000 <= CRVAL3 and
          float(wcs_coords[i*4+3])*1000 >= CRVAL3 + FREQ * -18518):
      #if (int(xyz_coords[i*7+6]) > 391 and int(xyz_coords[i*7+5]) < 409):
        zobs = float(words[i*10+2])
        wfreq = 1.4204E9/(1.0 + zobs) * float(words[i*10+5]) / 2.99E5
        self.true_set.append(
            { 'Entry':float(words[i*10]), 
            'Z_REAL':float(words[i*10+1]),
            'Z_OBS':zobs, 
            'RA':float(wcs_coords[i*4+1]),
            'DEC':float(wcs_coords[i*4+2]),
            'FREQ':float(wcs_coords[i*4+3])*1000,
            'w_FREQ':wfreq,
            'INCL':float(words[i*10+6]),
            'HIDIAM':float(words[i*10+7]),
            'HISIZE':float(words[i*10+8]),
            'INTFLUX':float(words[i*10+9]),
            'xs':int(xyz_coords[i*7+1]),
            'xe':int(xyz_coords[i*7+2]),
            'ys':int(xyz_coords[i*7+3]),
            'ye':int(xyz_coords[i*7+4]),
            'zs':int(xyz_coords[i*7+5]),
            'ze':int(xyz_coords[i*7+6]),
            'XW':(int(xyz_coords[i*7+2]) - int(xyz_coords[i*7+1])) * 0.00208333,
            'YW':(int(xyz_coords[i*7+4]) - int(xyz_coords[i*7+3])) * 0.00208333,
            'ZW':(int(xyz_coords[i*7+6]) - int(xyz_coords[i*7+5])),
            'X':(int(xyz_coords[i*7+2]) + int(xyz_coords[i*7+1])) / 2,
            'Y':(int(xyz_coords[i*7+4]) + int(xyz_coords[i*7+3])) / 2,
            'Z':(int(xyz_coords[i*7+6]) + int(xyz_coords[i*7+5])) / 2
            })

    self.true_set.sort(self.trueset_comparator)
    if (len(self.true_set) == 0):
      print ('no true positives in this dataset')
      exit()

    print ('There are ' + str(len(self.true_set)) + \
      ' true positives in this set')

    self.control_graphs = []
    self.best_completeness = {}  
    self.best_graphs = {}
    origin = 1420410000
    self.ZS = (origin - CRVAL3) / 18518
    print ('Starting frame ' + str(self.ZS))
    self.run_comparison(control_image, control_results_dir, 'control')

    # dictionary from jp2_param to best completeness achieved for that parameter
    # dictionary from jp2_param to graphs for best completeness achieved for
    # that parameter

  def parse_duchamp_file(self, file_name, control, atrous):
    # get to the important content
    words = open(file_name, 'r').read().split()
    i = 0 # word index offset to the source results
    prev = ''
    curr = words[i]
    while (i+2 < len(words) and (prev != '[]' or curr != '[pix]')):
      i+=1
      prev = curr
      curr = words[i]
      if (prev == 'detections' and curr == '='):
        detection_count = int(words[i+1])
    
    if (i >= len(words)):
      #TODO no soures found
      return
    i+=3 # get past underline '================'

    sources_found = []
    if (True):
      words_per_line = 39 #36
      for j in xrange(detection_count):
        RA_split = words[i + j*words_per_line + 5].split(':')
        DEC_split = words[i + j*words_per_line + 6].split(':')
        RA = float(RA_split[0])*15 + float(RA_split[1])/4 + \
            float(RA_split[2])/240
        DEC = float(DEC_split[0]) + float(DEC_split[1])/60 + \
            float(DEC_split[2])/3600
        sources_found.append (
            { 'ObjID':int(words[i + j*words_per_line]), \
              'Name':words[i + j*words_per_line + 1], \
              'X':float(words[i + j*words_per_line + 2]), \
              'Y':float(words[i + j*words_per_line + 3]), \
              'Z':float(words[i + j*words_per_line + 4])+self.ZS, \
              'RA':RA, \
              'DEC':DEC, \
              'VEL':self.freq_to_vel(float(words[i + j*words_per_line + 7])), \
              'MAJ':float(words[i + j*words_per_line + 8]), \
              'MIN':float(words[i + j*words_per_line + 9]), \
              'PA':float(words[i + j*words_per_line + 10]), \
              'w_RA':float(words[i + j*words_per_line + 11]), \
              'w_DEC':float(words[i + j*words_per_line + 12]), \
              'w_50':float(words[i + j*words_per_line + 13]), \
              'w_20':float(words[i + j*words_per_line + 14]), \
              'w_FREQ':float(words[i + j*words_per_line + 15]), \
              'INTFLUX':float(words[i + j*words_per_line + 16]), \
              'eINTFLUX':float(words[i + j*words_per_line + 17]), \
              'F_tot':float(words[i + j*words_per_line + 18]), \
              'eF_tot':float(words[i + j*words_per_line + 19]), \
              'F_peak':float(words[i + j*words_per_line + 20]), \
              'S/Nmax':float(words[i + j*words_per_line + 21]), \
              'X1':int(words[i + j*words_per_line + 22]), \
              'X2':int(words[i + j*words_per_line + 23]), \
              'Y1':int(words[i + j*words_per_line + 24]), \
              'Y2':int(words[i + j*words_per_line + 25]), \
              'Z1':int(words[i + j*words_per_line + 26]), \
              'Z2':int(words[i + j*words_per_line + 27]), \
              'XW':(int(words[i + j*words_per_line + 23]) -int(words[i +
                  j*words_per_line + 22])) * 0.00208333, \
              'YW':(int(words[i + j*words_per_line + 25]) -int(words[i +
                    j*words_per_line + 24])) * 0.00208333, \
              'ZW':(int(words[i + j*words_per_line + 27]) -int(words[i +
                  j*words_per_line + 26])), \
              'Npix':int(words[i + j*words_per_line + 28]), \
              'Flag':(words[i + j*words_per_line + 29]), \
              'X_av':float(words[i + j*words_per_line + 30]), \
              'Y_av':float(words[i + j*words_per_line + 31]), \
              'Z_av':float(words[i + j*words_per_line + 32]), \
              'X_cent':float(words[i + j*words_per_line + 33]), \
              'Y_cent':float(words[i + j*words_per_line + 34]), \
              'Z_cent':float(words[i + j*words_per_line + 35]), \
              'X_peak':float(words[i + j*words_per_line + 36]), \
              'Y_peak':float(words[i + j*words_per_line + 37]), \
              'Z_peak':float(words[i + j*words_per_line + 38]),
              'REDSHIFT':float(-1 + \
                  1420405751.77/float(words[i + j*words_per_line + 7])), \
              'FREQ':float(words[i + j*words_per_line + 7])
              })
    else:
      words_per_line = 36
      for j in xrange(detection_count):
        RA_split = words[i + j*words_per_line + 5].split(':')
        DEC_split = words[i + j*words_per_line + 6].split(':')
        RA = float(RA_split[0])*15 + float(RA_split[1])/4 + \
            float(RA_split[2])/240
        DEC = float(DEC_split[0]) + float(DEC_split[1])/60 + \
            float(DEC_split[2])/3600
        sources_found.append (
            { 'ObjID':int(words[i + j*words_per_line]), \
              'Name':words[i + j*words_per_line + 1], \
              'X':float(words[i + j*words_per_line + 2]), \
              'Y':float(words[i + j*words_per_line + 3]), \
              'Z':float(words[i + j*words_per_line + 4])+self.ZS, \
              'RA':RA, \
              'DEC':DEC, \
              'VEL':self.freq_to_vel(float(words[i + j*words_per_line + 7])), \
              'MAJ':float(words[i + j*words_per_line + 8]), \
              'MIN':float(words[i + j*words_per_line + 9]), \
              'PA':float(words[i + j*words_per_line + 10]), \
              'w_RA':float(words[i + j*words_per_line + 11]), \
              'w_DEC':float(words[i + j*words_per_line + 12]), \
              'w_50':float(words[i + j*words_per_line + 13]), \
              'w_20':float(words[i + j*words_per_line + 14]), \
              'w_FREQ':float(words[i + j*words_per_line + 15]), \
              'INTFLUX':float(words[i + j*words_per_line + 16]), \
              'XW':(int(words[i + j*words_per_line + 23]) -int(words[i +
                  j*words_per_line + 22])) * 0.00208333, \
              'YW':(int(words[i + j*words_per_line + 25]) -int(words[i +
                    j*words_per_line + 24])) * 0.00208333, \
              'ZW':(int(words[i + j*words_per_line + 27]) -int(words[i +
                  j*words_per_line + 26])), \
              'F_tot':float(words[i + j*words_per_line + 17]), \
              'F_peak':float(words[i + j*words_per_line + 18]), \
              'X1':int(words[i + j*words_per_line + 19]), \
              'X2':int(words[i + j*words_per_line + 20]), \
              'Y1':int(words[i + j*words_per_line + 21]), \
              'Y2':int(words[i + j*words_per_line + 22]), \
              'Z1':int(words[i + j*words_per_line + 23]), \
              'zs':int(words[i + j*words_per_line + 23]), \
              'Z2':int(words[i + j*words_per_line + 24]), \
              'ze':int(words[i + j*words_per_line + 24]), \
              'Npix':int(words[i + j*words_per_line + 25]), \
              'Flag':(words[i + j*words_per_line + 26]), \
              'X_av':float(words[i + j*words_per_line + 27]), \
              'Y_av':float(words[i + j*words_per_line + 28]), \
              'Z_av':float(words[i + j*words_per_line + 29]), \
              'X_cent':float(words[i + j*words_per_line + 30]), \
              'Y_cent':float(words[i + j*words_per_line + 31]), \
              'Z_cent':float(words[i + j*words_per_line + 32]), \
              'X_peak':float(words[i + j*words_per_line + 33]), \
              'Y_peak':float(words[i + j*words_per_line + 34]), \
              'Z_peak':float(words[i + j*words_per_line + 35]),
              'REDSHIFT':float(-1 + \
                  1420405751.77/float(words[i + j*words_per_line + 7])), \
              'FREQ':float(words[i + j*words_per_line + 7])
              })

    #sources_found.sort(self.trueset_comparator)
    return sources_found

  def run_comparison(self, image, results_dir, jp2_param):
    with open(os.devnull, 'w') as fnull: 
      subprocess.call(['mkdir', results_dir], stdout = fnull, stderr = fnull)

    duchamp_results_file = results_dir + '/duchamp'
 
    # Construct duchamp parameter file
    with open(results_dir + '/duc_params', "w+") as params:
      words = open('duchamp_parameters', 'r').read().split()
      for w in xrange(len(words)/2):
        print (words[w*2] + ' ' + words[w*2+1], file=params)
      print ('imageFile ' + image, file=params)
      print ('outFile ' + duchamp_results_file, file=params)
      #if (jp2_param == 'control'):
        #print ('flagATrous  1', file=params)

    with open(os.devnull, 'w') as fnull:
      subprocess.call([DUCHAMP, '-p',  results_dir + '/duc_params'], \
          stdout = fnull, stderr = fnull)
    
    atrous = True
    experiment = self.parse_duchamp_file (duchamp_results_file, jp2_param ==
        'control', atrous)
    print ('Duchamp identified ' + str(len(experiment)) + ' sources')
    print ('Compiling and Graphing results')
    match = {}
    already_identified = []

    # cross match
    for ti, t in enumerate(self.true_set):
      closest_match = sys.maxint
      matched_index = -1
      for ei, e in enumerate(experiment):
        if (( (t['xs'] <= e['X'] and e['X'] <= t['xe']
          and t['ys'] <= e['Y'] and e['Y'] <= t['ye']
          and t['zs'] <= e['Z'] and e['Z'] <= t['ze'])
        or (self.myabs(t['X'] - e['X']) <= 3 and
          self.myabs(t['Y'] - e['Y']) <= 3 and
          self.myabs(t['Z'] - e['Z']) <= 3) )
        and (self.myabs(t['X'] - e['X']) +
          self.myabs(t['Y'] - e['Y']) +
          self.myabs(t['Z'] - e['Z']) < closest_match)
          and not ei in already_identified):
            match[ti] = ei
            matched_index = ei
            closest_match = (e['X'] - t['xs']) - (t['xe'] - e['X']) + (e['Y'] - t['ys']) - (t['ye'] - e['Y']) + (e['Z'] - t['zs']) - (t['ze'] - e['Z'])
      if (not matched_index == -1):
        already_identified.append(matched_index)


    # Compile and graph statistics
    if (jp2_param == 'control'):
      print ("Control case identified " + str(len(match)) + " sources")
    else:
      print ("correctly identified " + str(len(match)) + " sources")
    if (len(match) == 0):
      target = open(results_dir + '/nomatches.txt', 'a')
      target.write("no matches of possible" + str(len(self.true_set)))
      if (jp2_param == 'control'):
        print ('WARNING: control found no sources')
        print ('=================================')
      print ("No matches found!")

    # Tally true and false positive results from Duchamp
    tp = len(match)
    fp = len(experiment) - len(match)
    fn = len(self.true_set) - len(match)

    if (jp2_param == 'control'):
      with open(results_dir + '/pos.txt', "w+") as posi:
        print ('tp ' + str(tp), file=posi)
        print ('fp ' + str(fp), file=posi)
        print ('fn ' + str(fn), file=posi)

    # calculate averages
    if (len(match) > 0):
      sums = { 'Z_OBS':0.0, 'w_FREQ':0.0, 'INCL':0.0, 'HIDIAM':0.0, 'HISIZE':0.0,
        'INTFLUX':0.0 }

      for m in match.keys():
        sums['Z_OBS'] += self.true_set[m]['Z_OBS']
        sums['w_FREQ'] += self.true_set[m]['w_FREQ']
        sums['INCL'] += self.true_set[m]['INCL']
        sums['HIDIAM'] += self.true_set[m]['HIDIAM']
        sums['HISIZE'] += self.true_set[m]['HISIZE']
        sums['INTFLUX'] += self.true_set[m]['INTFLUX']

      averages = {
        'Z_OBS':sums['Z_OBS']/len(match), 
        'w_FREQ':sums['w_FREQ']/len(match), 
        'INCL':sums['INCL']/len(match), 
        'HIDIAM':sums['HIDIAM']/len(match), 
        'HISIZE':sums['HISIZE']/len(match), 
        'INTFLUX':sums['INTFLUX']/len(match)
      }

      # calculate variances
      variances = { 'Z_OBS':0.0, 'w_FREQ':0.0, 'INCL':0.0, 'HIDIAM':0.0,
        'HISIZE':0.0, 'INTFLUX':0.0 }

      for m in match.keys():
        variances['Z_OBS'] += pow(self.true_set[m]['Z_OBS'] - averages['Z_OBS'], 2)
        variances['w_FREQ'] += pow(self.true_set[m]['w_FREQ'] - averages['w_FREQ'], 2)
        variances['INCL'] += pow(self.true_set[m]['INCL'] - averages['INCL'], 2)
        variances['HIDIAM'] += pow(self.true_set[m]['HIDIAM'] - averages['HIDIAM'], 2)
        variances['HISIZE'] += pow(self.true_set[m]['HISIZE'] - averages['HISIZE'], 2)
        variances['INTFLUX'] += pow(self.true_set[m]['INTFLUX'] -
            averages['INTFLUX'], 2)

      for v in variances.keys():
        variances[v] /= len(variances)

      # TODO: not sure how useful this currently is
      target = open(results_dir + '/averages.txt', 'a')
      target.write(str(averages))
      target = open(results_dir + '/variances.txt', 'a')
      target.write(str(variances))

    # identification percentage wrt. parameter value

    identified_set = deepcopy(self.true_set)
    for m in match.keys():
      identified_set[m]['match'] = 1
    steps = 20

    with open(results_dir + '/poster.txt', "w+") as post:
      for ident in identified_set:
        if ('match' in ident):
          print (str(ident['zs']) + ' ' + str(ident['ze']) + ' ' +
                str(ident['X']) + ' ' + str(ident['Y']), file=post)

    # Sets of Graphs

    # 0.  completeness and soundness of duchamp scan
    #     [completeness], [soundness]

    # 1.  'parameter' vs 'percentage identified'
    #     This is returned, to the parameter_exploration script. There
    #     this graph is overlayed with all other incremental steps
    #     exploring the respective parameter.
    #     [x. y. ylab, xlab, ylim, name]

    # 2.  RMSE of each parameter Duchamp found against the respective parameter
    #     in thetrueset.
    #     [rmse, name]

    # 3*. For the best combination of parameters found; RA-DEC error scatter
    #     plot. And Parameter error vs Count
    #     [not returned by this function]

    graphs = [0, 0, [],[]]

    # 0.
    graphs[0] = 1.0 * tp / (tp + fn) # completeness
    if (len(match) > 0):
      graphs[1] = 1.0 * tp / (tp + fp) # soundness
    else:
      graphs[1] = 0.0

    # 1.
    graphs[2].append (self.graph_perc_ident(identified_set, 'INTFLUX', steps,
          results_dir))
    graphs[2].append (self.graph_perc_ident(identified_set, 'w_FREQ', steps,
          results_dir))
    graphs[2].append (self.graph_perc_ident(identified_set, 'INCL', steps,
          results_dir))
    graphs[2].append (self.graph_perc_ident(identified_set, 'HIDIAM', steps,
          results_dir))

    # 2.
    graphs[3].append (self.rmse(match, experiment, 'RA'))
    graphs[3].append (self.rmse(match, experiment, 'Y'))
    graphs[3].append (self.rmse(match, experiment, 'FREQ'))
    graphs[3].append (self.rmse(match, experiment, 'INTFLUX'))
    graphs[3].append (self.rmse(match, experiment, 'w_FREQ'))
    graphs[3].append (self.rmse(match, experiment, 'XW'))
    graphs[3].append (self.rmse(match, experiment, 'YW'))

    # 3.
    self.scatter_plot(match, experiment, 'RA', 'DEC', results_dir)
    # eg - error graph
    ra_eg = self.error_graph(match, experiment, 'RA', results_dir)
    dec_eg = self.error_graph(match, experiment, 'DEC', results_dir)
    freq_eg = self.error_graph(match, experiment, 'FREQ', results_dir)
    intflux_eg = self.error_graph(match, experiment, 'INTFLUX', results_dir)
    wfreq_eg = self.error_graph(match, experiment, 'w_FREQ', results_dir)

    if (not jp2_param in self.best_completeness 
        or graphs[0] > self.best_completeness[jp2_param]):
      self.best_completeness[jp2_param] = graphs[0]
      # these best graphs are printed at the end overlayed with the control 
      # for easier comparison
      self.best_graphs[jp2_param] = [ra_eg, dec_eg, freq_eg, wfreq_eg]
      #, intflux_eg]

    if (jp2_param == 'control'):
      print ('completeness and soundness')
      print (graphs[0])
      print (graphs[1])
      self.control_match = match
      self.control_graphs = graphs
    return graphs

  def rmse(self, match, experiment, parameter):
    square_error_sum = 0.0
    count = 0
    for m in match.keys():
      count += 1

      square_error_sum += (experiment[match[m]][parameter] - \
         self.true_set[m][parameter])**2
    if (count == 0):
      return 0
    return math.sqrt(square_error_sum / count)

  def scatter_plot(self, match, experiment, param1, param2, results_dir):
    #TODO
    x = 'yolo'

  def error_graph(self, match, experiment, parameter, results_dir):
    if (len(match) == 0):
      return [[],[]]

    error = []
    for m in match.keys():
      error.append(experiment[match[m]][parameter] - \
          self.true_set[m][parameter])
    error.sort()
    steps = 20
    step_size = 1.0 * (error[-1] - error[0]) / (steps-1)
    error_steps = [0]*steps
    error_steps[0] = error[0]
    for e in xrange(len(error_steps)):
      if (not e == 0):
        error_steps[e] = error_steps[e-1] + step_size
    count = [0]*steps
    i = 0
    e = 0
    for s in error_steps:
      while (i < len(count) and e < len(error) and 
          error[e] < error_steps[i] + 0.5 * step_size):
        count[i] += 1
        e += 1
      i += 1 

    plt.clf()
    plt.plot(error_steps, count)
    plt.ylabel('Count')
    plt.xlabel(parameter + ' error')
    plt.savefig(results_dir + '/' + parameter + '_error.png')

    return [error_steps, count]

  def graph_perc_ident(self, identified_set, parameter, steps, results_dir):
    if (parameter == 'INTFLUX'):
      identified_set.sort(self.intflux_comparator)
    elif (parameter == 'w_FREQ'):
      identified_set.sort(self.wfreq_comparator)
    elif (parameter == 'INCL'):
      identified_set.sort(self.incl_comparator)
    elif (parameter == 'HIDIAM'):
      identified_set.sort(self.hdiam_comparator)
    step_size = float(identified_set[-1][parameter] - \
        identified_set[0][parameter])
    steps = 20
    step_size /= (steps)

    attempts = 0
    success = [0]*steps
    index = [0]*steps
    index_curr = identified_set[-1][parameter] - step_size
    for i in xrange(steps):
      if (i != 0):
        success[-i-1] = success[-i]*attempts
      index[-i-1] = index_curr

      #print (str(identified_set[-attempts-1][parameter]) + ' ' + str(index_curr))
      while (attempts < len(identified_set) and 
          identified_set[-attempts-1][parameter] + EPS >= index_curr):
        if ('match' in identified_set[-attempts-1]):
          success[-i-1] += 1.0
        attempts += 1
      if (attempts == 0):
        print ("ERROR: attempts = 0")
        exit()
      success[-i-1] /= attempts
      index_curr -= step_size

    return(index, success, 'Percentage identified', parameter, [0.0,1.0],
        parameter)

  def freq_to_vel(self, f):
    c = 299792458 # speed of light
    f0 = 1420405751.77 # HI frequency
    return c*( (f0-f) / f0 )

  def trueset_comparator(self, x, y):
    if (x['zs']-y['zs'] < 0):
      return -1
    else:
      return 1

  def intflux_comparator(self, x, y):
    if (x['INTFLUX']-y['INTFLUX'] < 0):
      return -1
    else:
      return 1

  def wfreq_comparator(self, x, y):
    if (x['w_FREQ']-y['w_FREQ'] < 0):
      return -1
    else:
      return 1

  def incl_comparator(self, x, y):
    if (x['INCL']-y['INCL'] < 0):
      return -1
    else:
      return 1

  def hdiam_comparator(self, x, y):
    if (x['HIDIAM']-y['HIDIAM'] < 0):
      return -1
    else:
      return 1

  def get_control_graphs(self):
    return self.control_graphs

  def myabs(self, x):
    if (x < 0):
      return -x
    else:
      return x
