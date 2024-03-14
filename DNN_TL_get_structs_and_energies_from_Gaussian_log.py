#!/usr/bin/env python3
# coding: utf-8

import os,sys
import time
os.system('bash -c "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate base" ')

current_path = os.getcwd()
if current_path[-1] == '/':
    current_path = current_path[0:-1]
user_name = current_path.split('/')[2]

os.chdir('{}/Edit_info_file/'.format( os.getcwd() ) )

input_information_get_cluster = 'DNN_TL_input_for_gjf.txt'
with open("{}".format(input_information_get_cluster),"r") as f:
     inf_line_get_cluster = f.readlines()

cluster_0                  = inf_line_get_cluster[1].split()[2]
os.chdir(current_path )
with open ('Run_YQ_DNN_TL_AUTO_Record.txt', 'a+') as f_re:
    f_re.write('Run time:{} python DNN_TL_run_get_gjf.py for {} \n'.format( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), cluster_0  ) )
    f_re.write('\n')

############################################ Define copy function ######################################################
# srcfile 需要复制、移动的文件   
# dstpath 目的地址
import glob
import shutil

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
       print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))
#######################################################################################################################

src_dir_info = '{}/Edit_info_file/'.format(current_path)

dst_dir_info = '{}/DNN_TL_{}/'.format(current_path, cluster_0)

src_file_list =  glob.glob(src_dir_info + '{}'.format(input_information_get_cluster) )

for srcfile in src_file_list:
    mycopyfile(srcfile, dst_dir_info)


input_information = '{}/DNN_TL_{}/DNN_TL_input_for_gjf.txt'.format( current_path, cluster_0  )
with open("{}".format(input_information),"r") as f:
     inf_line = f.readlines()

#****************************************** Cluster information *****************************************************

cluster                  = inf_line[1].split()[2]
if cluster[-1] == '-' or cluster[-1] == '+':
    cluster_name = cluster[0:-1]
else:
    cluster_name = cluster

atom_counts = defaultdict(int)
elements = re.findall(r'([A-Z][a-z]*)(\d*)', cluster)
print('elements:',elements)
for element, count in elements:
    if count == '':
        count = 1
    else:
        count = int(count)
    atom_counts[element] += count
atom_counts = dict(atom_counts)
print('atom_counts',atom_counts)
element = atom_counts.keys()
num_each_element = atom_counts.values()
num_atom_cluster = sum(num_each_element)

Min_Multi                = int(inf_line[2].split()[2])
Max_Multi                = int(inf_line[3].split()[2])
num_initial_structs_list = inf_line[4].split()[2].split(',')
num_structs_Min_Multi    = int(num_initial_structs_list[0])
num_structs_other_Multi  = int(num_initial_structs_list[1])
DFT_steps                = int(inf_line[5].split()[-1])
d_length                 = inf_line[6].split()[-1]
server_num               = inf_line[7].split()[2].split(',')
Guassian_tpye            = inf_line[8].split()[-1]
xc_function              = inf_line[9].split()[2]
basis_set                = inf_line[10].split()[2]
charge                   = int(inf_line[11].split()[-1])
key_word_list            = inf_line[12].split()[2:]
key_word                 = ' '.join(str(i) for i in key_word_list)
memory                   = inf_line[13].split()[-1]
nprocshared              = inf_line[14].split()[2]
iop_index                        = inf_line[15].split()[2]
if iop_index == 'True':
   iop = True 
else:
   iop = False

data_path = '{}/DNN_TL_{}/data'.format(current_path, cluster)
if not os.path.exists(data_path):
    os.mkdir( data_path  )

#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import shutil
#from sklearn import preprocessing
import sys, os, re, glob
sys.path.append(os.path.dirname(os.path.expanduser('~/Tools/')))
from avoid_overwritting import new_file_name

if len(element_row) == 1:
   from read_log_Gaussian import Read_log_Gaussian

if len(element_row) == 2:
   from read_log_Gaussian_MO import Read_log_Gaussian

if len(element_row) == 3:
   from read_log_Gaussian_MO import Read_log_Gaussian

if len(element_row) == 4:
       from read_log_Gaussian_MO import Read_log_Gaussian

if len(element_row) == 5:
       from read_log_Gaussian_MO import Read_log_Gaussian

from remove_abnormal_energy import Remove_abnormal_energy

pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100


# ### Write data file from log files

# In[2]:

for i in range( 0, int(0.5 * (Max_Multi-Min_Multi) + 1) ):
###### Where is the Gaussian output directory.
    dlogs = os.path.join(
            os.path.expandvars('$ACNNHOME'),
            '{}/DNN_TL_{}/Gaussian/{}/{}-steps-DFT-outputs/M{}'.format(current_path, cluster, cluster, DFT_steps, (Min_Multi + 2 * i)))
    dlogs = os.path.abspath(dlogs)
    dlogs = dlogs + '/*.log'
    logs = glob.iglob(dlogs)

###### Give a file name to write the structures.
    fout = os.path.join(
           os.path.expandvars('$ACNNHOME'),
           '{}/DNN_TL_{}/data/{}-M{}.xyz'.format(current_path, cluster, cluster, (Min_Multi + 2 * i)))
    fout = new_file_name(fout)

    for i, log in enumerate(logs):
        print('\n')
        print('The {}th log:'.format(i))
        print('The log path:{}'.format(log))
        popen_input_energy = 'grep "SCF Done" {} | wc -l'.format(log)
        ndone = int(os.popen(popen_input_energy).read())
        if ndone < 3:
            bad_log = '{}'.format(log)
            os.remove(bad_log)
            continue
        t = Read_log_Gaussian(log, element, num_atoms=num_atom_cluster, steps=DFT_steps, IOp= iop )
        if t.type in t.badtypelist:
           continue

        r = Remove_abnormal_energy(t.en, t.xyz)
        en = r.en
        xyz = r.xyz

        if len(element_row) == 1: 
           with open(fout, 'a+') as f:
               for j, (cords, e) in enumerate(zip(xyz, en)):
                   f.write('{}\n'.format(t.num_atoms))
                   f.write('STR:{}:{} = {:15.8f}\n'.format(i, j, e))
                   for cord in cords:
                       f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.
                               format(t.element, cord[0], cord[1], cord[2]))

        if len(element_row) == 2:
           num_first_atoms = num_each_element_1

           with open(fout, 'a+') as f:
               for j, (cords, e) in enumerate(zip(xyz, en)):
                   f.write('{}\n'.format(t.num_atoms))
                   f.write('STR:{}:{} = {:15.8f}\n'.format(i, j, e))
                   m = 1

                   for cord in cords:
                       if m <= num_first_atoms:
                           f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.
                                   format(t.element[0], cord[0], cord[1], cord[2]))
                           m = m + 1
                       elif m > num_first_atoms:
                           f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.
                                   format(t.element[1], cord[0], cord[1], cord[2]))

        if len(element_row) == 3:
           num_first_atoms = num_each_element_1
           num_second_atoms = (num_each_element_1 + num_each_element_2)

           with open(fout, 'a+') as f:
               for j, (cords, e) in enumerate(zip(xyz, en)):
                   f.write('{}\n'.format(t.num_atoms))
                   f.write('STR:{}:{} = {:15.8f}\n'.format(i, j, e))
                   m = 1

                   for cord in cords:
                      if m <= num_first_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'. 
                                  format(t.element[0], cord[0], cord[1], cord[2]))
                          m = m + 1   
                      elif num_second_atoms >=  m > num_first_atoms:        
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.
                                    format(t.element[1], cord[0], cord[1], cord[2]))
                          m = m + 1
                      elif m > num_second_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.
                                  format(t.element[2], cord[0], cord[1], cord[2]))
      
        if len(element_row) == 4:
           num_first_atoms = num_each_element_1
           num_second_atoms = (num_each_element_1 + num_each_element_2)
           num_third_atoms  = (num_each_element_1 + num_each_element_2 + num_each_element_3)

           with open(fout, 'a+') as f:
               for j, (cords, e) in enumerate(zip(xyz, en)):
                   f.write('{}\n'.format(t.num_atoms))
                   f.write('STR:{}:{} = {:15.8f}\n'.format(i, j, e))
                   m = 1

                   for cord in cords:
                      if m <= num_first_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[0], cord[0], cord[1], cord[2]))
                          m = m + 1
                      elif num_second_atoms >=  m > num_first_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[1], cord[0], cord[1], cord[2]))
                          m = m + 1
                      elif num_third_atoms >= m > num_second_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[2], cord[0], cord[1], cord[2]))
                          m = m + 1
                      elif m > num_third_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[3], cord[0], cord[1], cord[2]))
                          m = m + 1

        if len(element_row) == 5:
           num_first_atoms = num_each_element_1
           num_second_atoms = (num_each_element_1 + num_each_element_2)
           num_third_atoms  = (num_each_element_1 + num_each_element_2 + num_each_element_3)
           num_four_atoms  = (num_each_element_1 + num_each_element_2 + num_each_element_3 + num_each_element_4)
           with open(fout, 'a+') as f:
               for j, (cords, e) in enumerate(zip(xyz, en)):
                   f.write('{}\n'.format(t.num_atoms))
                   f.write('STR:{}:{} = {:15.8f}\n'.format(i, j, e))
                   m = 1
                   for cord in cords:
                      if m <= num_first_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[0], cord[0], cord[1], cord[2]))
                          m = m + 1                                                   
                      elif num_second_atoms >=  m > num_first_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[1], cord[0], cord[1], cord[2]))
                          m = m + 1
                      elif num_third_atoms >= m > num_second_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[2], cord[0], cord[1], cord[2]))
                          m = m + 1
                      elif num_four_atoms >= m > num_third_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[3], cord[0], cord[1], cord[2]))
                          m = m + 1
                      elif m > num_four_atoms:
                          f.write('{:<5s}{:15.8f}{:15.8f}{:15.8f}\n'.format(t.element[4], cord[0], cord[1], cord[2]))
                          m = m + 1




