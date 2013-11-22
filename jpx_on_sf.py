# AUTHOR: Sean Peters
# BRIEF: Script to compare the effects of the JPEG2000 compression pipeline
# on RMSE, compression ratio and source finding within the radio astronomy
# domain. The input file is a 1TB HDF5 cube. 
# DATE: 13 June 2013

from __future__ import print_function
from duchamp_comparator import duchamp_comparator
import subprocess
import sys
import os
import math
from jp2_parameter_explorer import jp2_parameter_explorer

def main():
  # Read in arguments
  if (len(sys.argv) == 6):
    results_dir = sys.argv[1] # where results will be saved
    control_image = sys.argv[2] # the fits cube being tested
    RA = sys.argv[3] # extent of RA dimension
    DEC = sys.argv[4] # extent of DEC dimension
    FREQ = sys.argv[5] # extent of FREQ dimension
  else:
    print ('results_dir control_image RA DEC FREQ')
    exit()

  trueset_file = 'dingo_cube.prt' # trueset of sources and their parameters

  # the starting frequency 

  CRVAL3 = 1413180000.0 #390
  #CRVAL3 = 1413000000.0 #400
  #CRVAL3 = 1412870000.0 #407
  #CRVAL3 = 1346330000 #4000
  #CRVAL3 = 1290780000 #7000
  #CRVAL3 = 1235220000 #10000
  #CRVAL3 =  1233370000 #10100
  #CRVAL3 = 1151890000.0 #14500
  #CRVAL3 = 1096330000 #17500

  # make directory where results will be produced
  with open(os.devnull, 'w') as fnull: 
    subprocess.call(['mkdir', results_dir], stdout = fnull, stderr = fnull)
  d = duchamp_comparator(trueset_file, control_image, results_dir + '/control/', \
      CRVAL3, float(FREQ))

  # each test compares the RMSE and/or Compression Ratio and/or Source finding
  # with the control, the constructor performs the control test as well
  jp2_explorer = jp2_parameter_explorer(RA, DEC, FREQ,  control_image, d)

  # JPEG2000 tests
  print ("Measuring JPX compression effect on Source Finding!")
  print ("===================================================")
  qstep_domain = [0.003, 0.01, 10, '*', 'log']
  jp2_explorer.run_test(results_dir + '/qstep/', 'Quantization step size', qstep_encoder,
      jp2_decoder, qstep_domain, 'qstep_encoded.jpx')

  clevels_domain = [1, 32, 4, '+', 'linear']
  jp2_explorer.run_test(results_dir + '/clevels/', 'Levels in DWT', clevels_encoder,
      jp2_decoder, clevels_domain, 'clev_encoded.jpx')

  precincts_domain = [64, 1024, 2, '*', 'log']
  jp2_explorer.run_test(results_dir + '/precincts/', 'Precinct Width (pixels)', precincts_encoder,
      jp2_decoder, precincts_domain, 'pre_encoded.jpx')

  cblk_domain = [4, 64, 2, '*', 'log']
  jp2_explorer.run_test(results_dir + '/cblks/', 'Block width (pixels)', cblk_encoder,
      jp2_decoder, cblk_domain, 'cblk_encoded.jpx')

  exit();

# these functions provide a mechanism for jp2_explorer to compress over each
# parameters respective parameter space

def qstep_encoder(i, input_cube, encoded_cube_name, ra, dec, freq):
  return ['/home/speters/work/SkuareView-NGAS-plugin/skuareview-encode', \
      '-i', input_cube, \
      '-o', encoded_cube_name, \
      '-icrop', '{0,0,0,' + ra + ',' + dec + ',' + freq + '}',
      'Qstep={' + ('%.10f' % i) + '}', 'Qderived={yes}']

def clevels_encoder(i, input_cube, encoded_cube_name, ra, dec, freq):
  return ['/home/speters/work/SkuareView-NGAS-plugin/skuareview-encode', \
      '-i', input_cube, \
      '-o', encoded_cube_name, \
      '-icrop', '{0,0,0,' + ra + ',' + dec + ',' + freq + '}',
      'Clevels={' + str(i) + '}']

def precincts_encoder(i, input_cube, encoded_cube_name, ra, dec, freq):
  ret = ['/home/speters/work/SkuareView-NGAS-plugin/skuareview-encode', \
      '-i', input_cube, \
      '-o', encoded_cube_name, \
      '-icrop', '{0,0,0,' + ra + ',' + dec + ',' + freq + '}']
  if (i == 64):
    ret.append ("Cprecincts={" + str(i) + "," + str(i) + "}")
  if (i == 128):
    ret.append ("Cprecincts={" + str(i) + "," + str(i) + "},{" + str(i/2) + \
        "," + str(i/2) + "}")
  if (i == 256):
    ret.append (prec_param = "Cprecincts={" + str(i) + "," + str(i) + "},{" + \
        str(i/2) + "," + str(i/2) + "},{" + str(i/4) + "," + str(i/4) + "}")
  if (i >= 512):
    ret.append ("Cprecincts={" + str(i) + "," + str(i) + "},{" + str(i/2) + \
        "," + str(i/2) + "},{" + str(i/4) + "," + str(i/4) + "},{" + \
        str(i/8) + "," + str(i/8) + "}")
  return ret

def cblk_encoder(i, input_cube, encoded_cube_name, ra, dec, freq):
  return ['/home/speters/work/SkuareView-NGAS-plugin/skuareview-encode', \
      '-i', input_cube, \
      '-o', encoded_cube_name, \
      '-icrop', '{0,0,0,' + ra + ',' + dec + ',' + freq + '}', \
      'Cblk={' + str(i) + ',' + str(i) + '}']

def jp2_decoder(encoded_cube_name, output_cube, ra, dec, freq):
  return ['/home/speters/work/SkuareView-NGAS-plugin/skuareview-decode', \
      '-i', encoded_cube_name, \
      '-o', output_cube]

main()
