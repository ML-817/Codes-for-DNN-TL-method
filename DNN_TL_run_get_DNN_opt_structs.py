#!/usr/bin python
# coding: utf-8

import os,re
import sys
import time
import numpy as np
from collections import defaultdict

current_path = os.getcwd()
if current_path[-1] == '/':
    current_path = current_path[0:-1]
user_name = current_path.split('/')[2]

os.chdir('{}/Edit_info_file/'.format( os.getcwd() ) )

input_information_get_cluster = 'DNN_TL_input_for_opt_structs.txt'

with open("{}".format(input_information_get_cluster),"r") as f:
     inf_line_get_cluster = f.readlines()

cluster_0                          = inf_line_get_cluster[2].split()[2]
print(cluster_0)
os.chdir(current_path )

with open ('Run_YQ_DNN_TL_AUTO_Record.txt', 'a+') as f_re:
    f_re.write('Run time:{} python DNN_TL_run_get_DNN_opt_stucts.py for {} \n'.format( time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), cluster_0  ) )
    f_re.write('\n')

################################################ Define mkdir function #################################################
# 创建安放初始结构文件夹
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False
# 定义要创建的目录
############################################# mkdir nn_fitting #########################################################
mkpath = "{}/DNN_TL_{}/nn_fitting".format(current_path, cluster_0)
# 调用函数
mkdir(mkpath)

mkpath = "{}/DNN_TL_{}/nn_fitting/Input_information_for_{}_opt_structs".format(current_path, cluster_0, cluster_0)
# 调用函数
mkdir(mkpath)

############################################# Define move function ####################################################
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("src not exist!")
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
######################################################################################################################        
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

src_file_list =  glob.glob(src_dir_info + 'DNN_TL_input_for_opt_structs.txt')

for srcfile in src_file_list:
    mycopyfile(srcfile, dst_dir_info)

input_information = '{}/DNN_TL_{}/DNN_TL_input_for_opt_structs.txt'.format( current_path, cluster_0  )
with open("{}".format(input_information),"r") as f:
     inf_line = f.readlines()

#****************************************** Cluster information *****************************************************
cluster                          = inf_line[2].split()[2]
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


offer_pretrain_cluster = inf_line[15].split()[2].split(',')
pre_atom_counts = defaultdict(int)
print(offer_pretrain_cluster)
print(type(offer_pretrain_cluster))
pre_elements = re.findall(r'([A-Z][a-z]*)(\d*)', offer_pretrain_cluster[0])
print('pre_elements:',pre_elements)
for pre_element, pre_count in pre_elements:
    if pre_count == '':
        pre_count = 1
    else:
        pre_count = int(pre_count)
    pre_atom_counts[pre_element] += pre_count
pre_atom_counts = dict(pre_atom_counts)
print('pre_atom_counts',pre_atom_counts)
pre_element = pre_atom_counts.keys()
pre_num_each_element = pre_atom_counts.values()
num_atom_pre_cluster = sum(pre_num_each_element)

Min_Multi                        = int(inf_line[3].split()[2])
Max_Multi                        = int(inf_line[4].split()[2])
num_initial_structs              = inf_line[5].split()[2]
d_length                         = inf_line[6].split()[-1]
server_num                       = inf_line[7].split()[2].split(',')

#***************************************** DNN information *********************************************************

cuda_num                         = inf_line[10].split()[2]
Refnn_epochs                     = inf_line[11].split()[2]
other_Multi_epochs               = inf_line[12].split()[2]
pick_lower_energy_num            = inf_line[13].split()[2]
Max_diff_for_filter              = inf_line[14].split()[2]
pretrain_cluster_Min_Multi       = int(inf_line[16].split()[2])

if cluster == offer_pretrain_cluster and num_atom_cluster == num_atom_pre_cluster :
   Transfer_learning_same  = 'NO'
   Transfer_learning_diff  = 'NO'
   print('!!!!! Transfer learning between different multiplicities of the cluster !!!!!!')   

if cluster != offer_pretrain_cluster and num_atom_cluster == num_atom_pre_cluster :
   Transfer_learning_same  = 'YES'
   Transfer_learning_diff  = 'NO'
   print('!!!!! Transfer learning between different clusters with the same number of atoms !!!!!!')

if cluster != offer_pretrain_cluster and num_atom_cluster != num_atom_pre_cluster :
   Transfer_learning_same  = 'NO'
   Transfer_learning_diff  = 'YES'
   print('!!!!! Transfer learning between different clusters with different number of atoms !!!!!!')    

#**************************************** Gaussian information *****************************************************

Guassian_tpye                    = inf_line[19].split()[-1]
xc_function                      = inf_line[20].split()[2]
basis_set                        = inf_line[21].split()[2]
DFT_steps                        = int(inf_line[22].split()[-1])
charge                           = int(inf_line[23].split()[-1])
key_word_list                    = inf_line[24].split()[2:]
key_word                         = ' '.join(str(i) for i in key_word_list)
memory                           = inf_line[25].split()[-1]
nprocshared                      = inf_line[26].split()[2]
basis_set_info_line              = 46 

#***************** Execute the switch of each function module 各功能模块的切换 ***************************************
Train_DNN                        = inf_line[30].split()[2]
Use_DNN_to_opt_structs           = inf_line[31].split()[2]
Creat_initial_structs            = inf_line[32].split()[2]
Pick_structs                     = inf_line[33].split()[2]
Filter                           = inf_line[34].split()[2]
Write_Full_steps_Gaussian_input  = inf_line[35].split()[2]
Run_Final_Min_Multi              = inf_line[36].split()[2]
Final_Min_Multi                  = inf_line[37].split()[2]
Multi_Transfer_to_Final          = inf_line[38].split()[2]
Transfer_from_Refnn              = inf_line[39].split()[2]
Refnn_Multi                      = inf_line[40].split()[2]
        
################################################ Copy creat_*_format files ############################################
src_dir = '{}/Format_file/'.format(current_path)

dst_dir = '{}/DNN_TL_{}/nn_fitting/'.format(current_path, cluster)

src_file_list =  glob.glob(src_dir + 'creat_*')

src_dir_1 = '{}/DNN_TL_{}/'.format(current_path, cluster)

dst_dir_1 = '{}/DNN_TL_{}/nn_fitting/Input_information_for_{}_opt_structs/'.format(current_path, cluster, cluster)

src_file_list_1  = glob.glob( src_dir_1 + '{}'.format(input_information_get_cluster) )
   
for srcfile in src_file_list:
    mycopyfile(srcfile, dst_dir)

for srcfile_1 in src_file_list_1:
    mycopyfile(srcfile_1, dst_dir_1)

os.rename('{}/DNN_TL_{}/nn_fitting/Input_information_for_{}_opt_structs/{}'.format(current_path, cluster, cluster, input_information_get_cluster), '{}/DNN_TL_{}/nn_fitting/Input_information_for_{}_opt_structs/{}_{}'.format(current_path, cluster, cluster, input_information_get_cluster, time.strftime("%Y-%m-%d-%H_%M", time.localtime())))
################################################ mkdir Refnn-M{} #######################################################
if (cluster != offer_pretrain_cluster) or (pretrain_cluster_Min_Multi == Min_Multi and cluster == offer_pretrain_cluster): 
   mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}".format(current_path, cluster, cluster, Min_Multi)
   mkdir(mkpath)
########################################################################################################################

################################################ Define counter files ##################################################
def Counter_files():
    path = os.getcwd()    #获取当前路径
    count = 0
    for root,dirs,files in os.walk(path):    #遍历统计
          for each in files:
                 count += 1   #统计文件夹下文件个数
    return count               #输出结果
#######################################################################################################################


#################################### Define Creat_get_pretrain_network function #######################################
def Creat_get_pretrain_network_json(current_path, cluster, offer_pretrain_cluster, pretrain_cluster_Min_Multi, cuda_num, Transfer_learning_diff, Min_Multi ):

    mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-from-{}-M{}".format(current_path, cluster, cluster, offer_pretrain_cluster, pretrain_cluster_Min_Multi )
    mkdir(mkpath)

    src_dir = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/".format(current_path, offer_pretrain_cluster, offer_pretrain_cluster, pretrain_cluster_Min_Multi)

    dst_dir = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-from-{}-M{}/".format(current_path, cluster, cluster, offer_pretrain_cluster, pretrain_cluster_Min_Multi )

    src_file_list =  glob.glob(src_dir + '*')

    for srcfile in src_file_list:
        mycopyfile(srcfile, dst_dir)

    Creat_get_pretrain_network_json='{}/DNN_TL_{}/nn_fitting/{}-fit-get_pretrain_network.json'.format(current_path, cluster, cluster)
    creat_Refnn_network_format_json='{}/DNN_TL_{}/nn_fitting/creat_Refnn_network_format.json'.format(current_path, cluster)
    lines_input_1 = ''
    with open(creat_Refnn_network_format_json, 'r') as fread:
        for line in fread.readlines()[0:7]:  # in this case, let's copy line 0-6
                lines_input_1 += line

    lines_input_2 = ''
    with open(creat_Refnn_network_format_json, 'r') as fread:
        for line in fread.readlines()[10:28]:  # in this case, let's copy line 0-6
                lines_input_2 += line

    lines_input_3 = ''
    with open(creat_Refnn_network_format_json, 'r') as fread:
        for line in fread.readlines()[30:40]:  # in this case, let's copy line 0-6
                lines_input_3 += line

    lines_input_4 = ''
    with open(creat_Refnn_network_format_json, 'r') as fread:
        for line in fread.readlines()[41:]:  # in this case, let's copy line 0-6
                lines_input_4 += line

    with open (Creat_get_pretrain_network_json, 'w') as f:
        f.truncate()
    with open (Creat_get_pretrain_network_json, 'a+') as f:
        f.write(lines_input_1)
        f.write('    "output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}",\n'.format(current_path, cluster, cluster, cluster))
        f.write('    "gpu": "cuda{}",\n'.format(cuda_num))
        f.write('    "Transfer_learning_between_different_clusters":"{}",\n'.format(Transfer_learning_diff))
        f.write(lines_input_2)
        f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_network.dill.0",\n'.format(current_path, offer_pretrain_cluster, offer_pretrain_cluster, pretrain_cluster_Min_Multi))
        f.write('\t"epochs": {},\n'.format(1))
        f.write(lines_input_3)
        f.write('\t"input_file": "{}/DNN_TL_{}/data/{}-M{}.xyz.0",\n'.format(current_path, cluster, cluster, Min_Multi))
        f.write(lines_input_4)
###################################################################################################################


#################################### Define Run_get_pretrain_network function #########################################
def Run_get_pretrain_network_json( current_path, cluster ):

      mkpath_1 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}".format(current_path, cluster, cluster, cluster)
      mkdir(mkpath_1)

      mkpath_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/bad-network".format(current_path, cluster, cluster, cluster )
      mkdir(mkpath_2)
         
      os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))

      print('****************************************************************************************************')
      for i in range(0,6):
          print('********************************* Run get_pretrain_network.json ************************************')
      print('****************************************************************************************************')


      os.system('acnnmain {}-fit-get_pretrain_network.json'.format(cluster))
      fit_error_txt = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/fit_error.txt.0".format(current_path, cluster, cluster, cluster)
      counter_Pretrain_nn = 0
      Min_Pretrain_nn_error_index = counter_Pretrain_nn
      with open(fit_error_txt,"r") as f:
           inf_line = f.readlines()
      
      if float(inf_line[0].split()[-2]) > 0.45 :
         counter_Pretrain_nn = counter_Pretrain_nn + 1

         os.system('acnnmain {}-fit-get_pretrain_network.json'.format(cluster))
         fit_error_txt_1 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/fit_error.txt.1".format(current_path, cluster, cluster, cluster)

         with open(fit_error_txt_1,"r") as f:
              inf_line_1 = f.readlines()
         if float(inf_line_1[0].split()[-2]) > 0.45 :
            counter_Pretrain_nn = counter_Pretrain_nn + 1

            print('****************************************************************************************************')
            for i in range(0,6):
                print('********************************* Run the 2nd get_pretrain_network.json ************************************')
            print('****************************************************************************************************')
            os.system('acnnmain {}-fit-get_pretrain_network.json'.format(cluster))
            fit_error_txt_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/fit_error.txt.2".format(current_path, cluster, cluster, cluster)

            with open(fit_error_txt_2,"r") as f:
                 inf_line_2 = f.readlines()

            if float(inf_line_2[0].split()[-2]) > 0.45 :

               print('***************************************************************************************************')
               for i in range(0,6):
                   print('********************************* Run the 3nd get_pretrain_network.json ************************************')
               print('***************************************************************************************************')
               mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/bad-network".format(current_path, cluster, cluster, cluster )
               shutil.rmtree(mkpath)
               mkdir(mkpath)
               des_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/bad-network/".format(current_path, cluster, cluster, cluster )
               Bad_pretrain_nn_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/".format(current_path, cluster, cluster, cluster)      
               src_file_list =  glob.glob( Bad_pretrain_nn_path + '*' )

               for srcfile in src_file_list:
                   mymovefile(srcfile, des_path)
                  
               print('These three Pretrain-DNN testing errors are all > 0.35 ev ! !!!!Please find the reason!!!!')
               sys.exit()

      Min_Pretrain_nn_error_index = counter_Pretrain_nn
      return Min_Pretrain_nn_error_index
###################################################################################################################


#################################### Define Creat Refnn_network.json function #########################################
def Creat_Refnn_network_json( current_path, cluster, Min_Multi, cuda_num, offer_pretrain_cluster, pretrain_cluster_Min_Multi, Min_Pretrain_nn_error_index, Refnn_epochs, Transfer_learning_diff ):
   Creat_refnn_network_json='{}/DNN_TL_{}/nn_fitting/{}-fit-Refnn_network.json'.format(current_path, cluster, cluster)
   creat_Refnn_network_format_json='{}/DNN_TL_{}/nn_fitting/creat_Refnn_network_format.json'.format(current_path, cluster)
   lines_input_1 = ''
   with open(creat_Refnn_network_format_json, 'r') as fread:
       for line in fread.readlines()[0:7]:  # in this case, let's copy line 0-6
               lines_input_1 += line

   lines_input_2 = ''
   with open(creat_Refnn_network_format_json, 'r') as fread:
       for line in fread.readlines()[10:28]:  # in this case, let's copy line 0-6
               lines_input_2 += line

   lines_input_3 = ''
   with open(creat_Refnn_network_format_json, 'r') as fread:
       for line in fread.readlines()[30:40]:  # in this case, let's copy line 0-6
               lines_input_3 += line

   lines_input_4 = ''
   with open(creat_Refnn_network_format_json, 'r') as fread:
       for line in fread.readlines()[41:]:  # in this case, let's copy line 0-6
               lines_input_4 += line

   with open (Creat_refnn_network_json, 'w') as f:
       f.truncate()
   with open (Creat_refnn_network_json, 'a+') as f:
       f.write(lines_input_1)
       f.write('    "output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}",\n'.format(current_path, cluster, cluster, Min_Multi))
       f.write('    "gpu": "cuda{}",\n'.format(cuda_num))
       f.write('    "Transfer_learning_between_different_clusters":"NO",\n')
       f.write(lines_input_2)
       if Transfer_learning_same == 'YES' or (cluster == offer_pretrain_cluster and pretrain_cluster_Min_Multi != Min_Multi) :
          f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_network.dill.0",\n'.format(current_path, offer_pretrain_cluster, offer_pretrain_cluster, pretrain_cluster_Min_Multi))
       elif Transfer_learning_diff == 'YES':
          f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Pretrain-nn-for-{}/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, cluster, Min_Pretrain_nn_error_index ))
       elif cluster == offer_pretrain_cluster and pretrain_cluster_Min_Multi == Min_Multi :
          f.write('\t"load_network": -1,\n')
       f.write('\t"epochs": {},\n'.format(Refnn_epochs))
       f.write(lines_input_3)
       f.write('\t"input_file": "{}/DNN_TL_{}/data/{}-M{}.xyz.0",\n'.format(current_path, cluster, cluster, Min_Multi))
       f.write(lines_input_4)
#######################################################################################################################


####################################### Define Run_Refnn_network function ##############################################
def Run_Refnn_network_json( current_path, cluster, Min_Multi ):
   os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))

   print('*********************************************************************************************')
   for i in range(0,6):
       print('********************************* Run Refnn_network.json ************************************')
   print('*********************************************************************************************')

   os.system('acnnmain {}-fit-Refnn_network.json'.format(cluster))
   fit_error_txt = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/fit_error.txt.0".format(current_path, cluster, cluster, Min_Multi)
   counter_Refnn = 0
   Min_Refnn_error_index = counter_Refnn
   with open(fit_error_txt,"r") as f:
        inf_line = f.readlines()
   count=int(len(open(fit_error_txt,'rU').readlines()))
   list1 = []
   list2 = []
   training_error_list = []
   for ii in range(-count/2,-count/4):
       list1.append(float(inf_line[ii].split()[-2]))
   for jj in range(-count/2,-1):
       list2.append(float(inf_line[jj].split()[-2]))
   for kk in range(-count/2,-1):
       training_error_list.append(float(inf_line[kk].split()[-3]))
   valid_error_1 = min(list1)
   valid_error_2 = min(list2)
   valid_avg_error_1 = np.mean(list1)
   valid_avg_error_2 = np.mean(list2)
   train_avg_error   = np.mean(training_error_list)
   if (float(inf_line[0].split()[-2]) > 0.4) or (float(inf_line[0].split()[-2]) > 0.3 and valid_error_1 < valid_error_2) or (27.2114*(valid_avg_error_1 - valid_avg_error_2) > 0.3) or (27.2114*(valid_avg_error_2 - train_avg_error) > 0.5) :

      print('*********************************************************************************************')
      for i in range(0,6):
          print('********************************* Run the 2nd Refnn_network.json ************************************')
      print('*********************************************************************************************')

      os.system('acnnmain {}-fit-Refnn_network.json'.format(cluster))
      counter_Refnn = counter_Refnn + 1

      fit_error_txt_1 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/fit_error.txt.1".format(current_path, cluster, cluster, Min_Multi)

      with open(fit_error_txt_1,"r") as f:
           inf_line_1 = f.readlines()
      count=int(len(open(fit_error_txt,'rU').readlines()))
      list1 = []
      list2 = []
      training_error_list = []
      for ii in range(-count/2,-count/4):
          list1.append(float(inf_line_1[ii].split()[-2]))
      for jj in range(-count/2,-1):
          list2.append(float(inf_line_1[jj].split()[-2]))
      for kk in range(-count/2,-1):
          training_error_list.append(float(inf_line_1[kk].split()[-3]))
      valid_error_1 = min(list1)
      valid_error_2 = min(list2)
      valid_avg_error_1 = np.mean(list1)
      valid_avg_error_2 = np.mean(list2)
      train_avg_error   = np.mean(training_error_list)

      if (float(inf_line[0].split()[-2]) > 0.4) or (float(inf_line_1[0].split()[-2]) > 0.35 and valid_error_1 < valid_error_2) or (27.2114*(valid_avg_error_1 - valid_avg_error_2) > 0.3) or (27.2114*(valid_avg_error_2 - train_avg_error) > 0.5) :

         print('*********************************************************************************************')
         for i in range(0,6):
             print('********************************* Run the 3nd Refnn_network.json ************************************')
         print('*********************************************************************************************')

         os.system('acnnmain {}-fit-Refnn_network.json'.format(cluster))
         counter_Refnn = counter_Refnn + 1

         fit_error_txt_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/fit_error.txt.2".format(current_path, cluster, cluster, Min_Multi)

         with open(fit_error_txt_2,"r") as f:
              inf_line_2 = f.readlines()
         count=int(len(open(fit_error_txt,'rU').readlines()))
         list1 = []
         list2 = []
         training_error_list = []
         for ii in range(-count/2,-count/4):
             list1.append(float(inf_line_2[ii].split()[-2]))
         for jj in range(-count/2,-1):
             list2.append(float(inf_line_2[jj].split()[-2]))
         for kk in range(-count/2,-1):
             training_error_list.append(float(inf_line_2[kk].split()[-3]))
         valid_error_1 = min(list1)
         valid_error_2 = min(list2)
         valid_avg_error_1 = np.mean(list1)
         valid_avg_error_2 = np.mean(list2)
         train_avg_error   = np.mean(training_error_list)

         if (float(inf_line[0].split()[-2]) > 0.4) or (float(inf_line_2[0].split()[-2]) > 0.37 and valid_error_1 < valid_error_2) or (27.2114*(valid_avg_error_1 - valid_avg_error_2) > 0.3) or (27.2114*(valid_avg_error_2 - train_avg_error) > 0.6) :
            mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/bad-network".format(current_path, cluster, cluster, Min_Multi )
            mkdir(mkpath)
            shutil.rmtree(mkpath)
            mkdir(mkpath)
            des_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/bad-network/".format(current_path, cluster, cluster, Min_Multi )
            Bad_pretrain_nn_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}".format(current_path, cluster, cluster, Min_Multi )
            src_file_list =  glob.glob( Bad_pretrain_nn_path + '*' )
            for srcfile in src_file_list:
                mymovefile(srcfile, des_path)

            print('These three Ref-DNN testing errors are all > 0.3 ev ! !!!!Please find the reason!!!!')
            sys.exit()

   Min_Refnn_error_index = counter_Refnn
   return Min_Refnn_error_index
########################################################################################################################

#################################### Define Creat and Run creat.json function #########################################
def Creat_and_Run_creat_json( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, cluster_name, num_initial_structs, d_length, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi):
          os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))

          if int(Final_Min_Multi) == int(Present_Multi) and Run_Final_Min_Multi == 'ON': 
             mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-gas-Final-M{}".format(current_path, cluster, cluster, Present_Multi)
             Creat_file = '{}/DNN_TL_{}/nn_fitting/{}-Final-M{}-create.json'.format(current_path, cluster, cluster, Present_Multi)
          else:
             mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-gas-M{}".format(current_path, cluster, cluster, Present_Multi)
             Creat_file = '{}/DNN_TL_{}/nn_fitting/{}-M{}-create.json'.format(current_path, cluster, cluster, Present_Multi)
          mkdir(mkpath)

          with open (Creat_file, 'w') as f:
              f.truncate()
          with open (Creat_file, 'a+') as f:
              f.write('{\n')
              f.write('"tasks": ["create"],\n')
              if  int(Final_Min_Multi) == int(Present_Multi) and Run_Final_Min_Multi == 'ON':
                 f.write('"output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-gas-Final-M{}",\n'.format(current_path, cluster, cluster, Present_Multi))
              else:
                 f.write('"output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-gas-M{}",\n'.format(current_path, cluster, cluster, Present_Multi))
              f.write('"random_seed": 0,\n')
              f.write('"creation": {\n')
              f.write('"name": "{}",\n'.format(cluster_name))
              f.write('"number": {},\n'.format(num_initial_structs))
              f.write('"method": "blda",\n')
              f.write('"order": 2,\n')
              f.write('"2d": {}\n'.format(d_length))
              f.write('\t},\n')
              f.write('"filtering-create": {"max_diff": 0.25,"max_diff_report": 1.00}\n')
              f.write('}')
          print('*********************************************************************************************')
          for k in range(0,6):
              if int(Final_Min_Multi) == int(Present_Multi) and Run_Final_Min_Multi == 'NO' :
                 print('********************************* Run {}-Final-M{}-creat.json ************************************'.format(cluster, Present_Multi))
              else:
                 print('********************************* Run {}-M{}-creat.json ************************************'.format(cluster, Present_Multi))
          print('*********************************************************************************************')
          if int(Final_Min_Multi) == int(Present_Multi) and Run_Final_Min_Multi == 'ON' :

             os.system('acnnmain {}-Final-M{}-create.json'.format(cluster, Present_Multi))
          else:
             os.system('acnnmain {}-M{}-create.json'.format(cluster, Present_Multi))
#######################################################################################################################

##################################### Define Creat Other_networks.json function #######################################
def Creat_Other_networks_json( Max_Multi, Min_Multi, current_path, cluster, cuda_num, offer_pretrain_cluster, pretrain_cluster_Min_Multi, other_Multi_epochs, Min_Refnn_error_index, Min_M_error_index, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi, Transfer_from_Refnn, Refnn_Multi, Multi_Transfer_to_Final ): 

       if int(Final_Min_Multi) == int(Present_Multi) and Run_Final_Min_Multi == 'ON' :
          Creat_M_network_json='{}/DNN_TL_{}/nn_fitting/{}-fit-Final-M{}_network.json'.format( current_path, cluster, cluster,  Present_Multi )
       else:
          Creat_M_network_json='{}/DNN_TL_{}/nn_fitting/{}-fit-M{}_network.json'.format( current_path, cluster, cluster,  Present_Multi )

       creat_M_network_format_json='{}/DNN_TL_{}/nn_fitting/creat_other_networks_format.json'.format(current_path, cluster)
       lines_input_1 = ''
       with open(creat_M_network_format_json, 'r') as fread:
           for line in fread.readlines()[0:7]:  # in this case, let's copy line 0-7
                   lines_input_1 += line

       lines_input_2 = ''
       with open(creat_M_network_format_json, 'r') as fread:
           for line in fread.readlines()[10:28]:  # in this case, let's copy line 0-6
                   lines_input_2 += line

       lines_input_3 = ''
       with open(creat_M_network_format_json, 'r') as fread:
           for line in fread.readlines()[30:40]:  # in this case, let's copy line 0-6
                   lines_input_3 += line

       lines_input_4 = ''
       with open(creat_M_network_format_json, 'r') as fread:
           for line in fread.readlines()[41:]:  # in this case, let's copy line 0-6
                   lines_input_4 += line

       with open (Creat_M_network_json, 'w') as f:
           f.truncate()
       with open (Creat_M_network_json, 'a+') as f:
           f.write(lines_input_1)
           if int(Final_Min_Multi) == int(Present_Multi) and Run_Final_Min_Multi == 'ON' :
              f.write('    "output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn",\n'.format(current_path, cluster,cluster, Present_Multi))
           else:
              f.write('    "output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn",\n'.format(current_path, cluster,cluster, Present_Multi))
           f.write('    "gpu": "cuda{}",\n'.format(cuda_num))
           f.write('    "Transfer_learning_between_different_clusters":"NO",\n')
           f.write(lines_input_2)
           if Transfer_from_Refnn == 'ON' and int(Refnn_Multi) == int(int(Present_Multi) - 2) :
              f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Refnn_Multi, Min_Refnn_error_index))

           elif Run_Final_Min_Multi == 'ON' and int(Final_Min_Multi) == int(Present_Multi) and Min_Multi != Max_Multi :   
              f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Max_Multi, Min_Refnn_error_index))

           elif Run_Final_Min_Multi == 'ON' and int(Final_Min_Multi) == int(Present_Multi) and Min_Multi == Max_Multi :
              f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Multi_Transfer_to_Final, Min_Refnn_error_index))              

           elif Transfer_from_Refnn == 'OFF' and Run_Final_Min_Multi == 'OFF' and int(Min_Multi) == int(Present_Multi) :              f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.0",\n'.format(current_path, cluster, cluster, pretrain_cluster_Min_Multi )) 

           else:   
              f.write('\t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, (Present_Multi - 2), Min_M_error_index ))
           f.write('\t"epochs": {},\n'.format(other_Multi_epochs))
           f.write(lines_input_3)
           if Run_Final_Min_Multi == 'ON' and int(Final_Min_Multi) == int(Present_Multi) :
              f.write('\t"input_file": "{}/DNN_TL_{}/data/{}-M{}.xyz.0",\n'.format(current_path, cluster, cluster, Present_Multi))
           else:
              f.write('\t"input_file": "{}/DNN_TL_{}/data/{}-M{}.xyz.0",\n'.format(current_path, cluster, cluster, Present_Multi))
           f.write(lines_input_4)
########################################################################################################################



##################################### Define Run Other network.json function ###########################################
def Run_Other_network_json( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi, Transfer_from_Refnn, Refnn_Multi ):
   os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))

   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
       mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn".format(current_path, cluster, cluster, Present_Multi)
   else:
       mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn".format(current_path, cluster, cluster, Present_Multi)
   mkdir(mkpath)

   print('*********************************************************************************************')
   for k in range(0,6):
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi)  :
          print('********************************* Run {}-fit-Final-M{}_network.json ************************************'.format(cluster, Present_Multi)) 
       else:
          print('********************************* Run {}-fit-M{}_network.json ************************************'.format(cluster, Present_Multi))
   print('*********************************************************************************************')
   
   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi)  :
      os.system('acnnmain {}-fit-Final-M{}_network.json'.format(cluster, Present_Multi))
   else:
      os.system('acnnmain {}-fit-M{}_network.json'.format(cluster, Present_Multi))

   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
      fit_error_txt = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_error.txt.0".format(current_path, cluster, cluster, Present_Multi)
   else:
      fit_error_txt = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_error.txt.0".format(current_path, cluster, cluster, Present_Multi)
   counter_M = 0
   Min_M_error_index = counter_M

   with open(fit_error_txt,"r") as f:
       inf_line = f.readlines()
   count=int(len(open(fit_error_txt,'rU').readlines()))
   list1 = []
   list2 = []
   training_error_list = []
   for ii in range(-count/2,-count/4):
       list1.append(float(inf_line[ii].split()[-2]))
   for jj in range(-count/4,-1):
       list2.append(float(inf_line[jj].split()[-2]))
   for kk in range(-count/4,-1):
       training_error_list.append(float(inf_line[kk].split()[-3]))

   valid_error_1 = min(list1)
   valid_error_2 = min(list2)
   valid_avg_error_1 = np.mean(list1)
   valid_avg_error_2 = np.mean(list2)
   train_avg_error   = np.mean(training_error_list)

   if (float(inf_line[0].split()[-2]) > 0.32) or (float(inf_line[0].split()[-2]) > 0.31 and valid_error_1 < valid_error_2) or (27.2114*(valid_avg_error_1 - valid_avg_error_2) > 0.3) or (27.2114*(valid_avg_error_2 - train_avg_error) > 0.4) :
      print('*********************************************************************************************')
      for k in range(0,6):
          if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
             print('********************************* Run the 2nd {}-fit-Final-M{}_network.json ************************************'.format(cluster, Present_Multi))
          else:
             print('********************************* Run the 2nd {}-fit-M{}_network.json ************************************'.format(cluster, Present_Multi))
      print('*********************************************************************************************')
      
      if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi)  :
         os.system('acnnmain {}-fit-Final-M{}_network.json'.format(cluster, Present_Multi ))
      else:
         os.system('acnnmain {}-fit-M{}_network.json'.format(cluster, Present_Multi ))
      counter_M = counter_M + 1

      if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
         fit_error_txt_1 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_error.txt.1".format(current_path, cluster, cluster, Present_Multi)
      else:
         fit_error_txt_1 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_error.txt.1".format(current_path, cluster, cluster, Present_Multi)

      with open(fit_error_txt_1,"r") as f:
          inf_line_1 = f.readlines()
      count=int(len(open(fit_error_txt,'rU').readlines()))
      list1 = []
      list2 = []
      training_error_list = []
      for ii in range(-count/2,-count/4):
          list1.append(float(inf_line_1[ii].split()[-2]))
      for jj in range(-count/4,-1):
          list2.append(float(inf_line_1[jj].split()[-2]))
      for kk in range(-count/4,-1):
          training_error_list.append(float(inf_line_1[kk].split()[-3]))
      valid_error_1 = min(list1)
      valid_error_2 = min(list2)
      valid_avg_error_1 = np.mean(list1)
      valid_avg_error_2 = np.mean(list2)
      train_avg_error   = np.mean(training_error_list)

      if (float(inf_line[0].split()[-2]) > 0.34) or (float(inf_line_1[0].split()[-2]) > 0.33 and valid_error_1 < valid_error_2) or (27.2114*(valid_avg_error_1 - valid_avg_error_2) > 0.3) or (27.2114*(valid_avg_error_2 - train_avg_error) > 0.5) :
         print('*********************************************************************************************')
         for k in range(0,6):
             if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
                print('********************************* Run the 3nd {}-fit-Final-M{}_network.json ************************************'.format(cluster, Present_Multi))
             else:
                print('********************************* Run the 3nd {}-fit-M{}_network.json ************************************'.format(cluster, Present_Multi))
         print('*********************************************************************************************')

         if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
            os.system('acnnmain {}-fit-Final-M{}_network.json'.format(cluster, Present_Multi))
         else:
            os.system('acnnmain {}-fit-M{}_network.json'.format(cluster, Present_Multi))
         counter_M = counter_M + 1
         if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
            fit_error_txt_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_error.txt.2".format(current_path, cluster, cluster, Present_Multi)
         else:
            fit_error_txt_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_error.txt.2".format(current_path, cluster, cluster, Present_Multi)

         with open(fit_error_txt_2,"r") as f:
              inf_line_2 = f.readlines()
         count=int(len(open(fit_error_txt,'rU').readlines()))
         list1 = []
         list2 = []
         training_error_list = []
         for ii in range(-count/2,-count/4):
             list1.append(float(inf_line_2[ii].split()[-2]))
         for jj in range(-count/4,-1):
             list2.append(float(inf_line_2[jj].split()[-2]))
         for kk in range(-count/4,-1):
             training_error_list.append(float(inf_line_2[kk].split()[-3]))
         valid_error_1 = min(list1)
         valid_error_2 = min(list2)
         valid_avg_error_1 = np.mean(list1)
         valid_avg_error_2 = np.mean(list2)
         train_avg_error   = np.mean(training_error_list)

         if (float(inf_line[0].split()[-2]) > 0.35) or (float(inf_line_2[0].split()[-2]) > 0.34 and valid_error_1 < valid_error_2) or (27.2114*(valid_avg_error_1 - valid_avg_error_2) > 0.3) or (27.2114*(valid_avg_error_2 - train_avg_error) > 0.5) :
            if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
               mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/bad-network".format(current_path, cluster, cluster, Present_Multi )
            elif Transfer_from_Refnn == 'ON' and int(Final_Min_Multi) == int(Present_Multi) :
               mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/bad-network".format(current_path, cluster, cluster, Present_Multi )

            else:
               mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/bad-network".format(current_path, cluster, cluster, Present_Multi )
            mkdir(mkpath)
            shutil.rmtree(mkpath)
            mkdir(mkpath)
            if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
               des_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/bad-network/".format(current_path, cluster, cluster, Present_Multi )
               Bad_pretrain_nn_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn".format(current_path, cluster, cluster, Present_Multi )

            elif Transfer_from_Refnn == 'ON' and int(Final_Min_Multi) == int(Present_Multi) :
               des_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}/bad-network/".format(current_path, cluster, cluster, Present_Multi ) 
               Bad_pretrain_nn_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Refnn-M{}".format(current_path, cluster, cluster, Present_Multi )

            else:
               des_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/bad-network/".format(current_path, cluster, cluster, Present_Multi )
               Bad_pretrain_nn_path = "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn".format(current_path, cluster, cluster, Present_Multi )

            src_file_list =  glob.glob( Bad_pretrain_nn_path + '*' )
            for srcfile in src_file_list:
                mymovefile(srcfile, des_path)

            print('These three DNN testing errors are all > 0.25 ev ! !!!!Please find the reason!!!!')
            sys.exit()

   Min_M_error_index = counter_M
   return Min_M_error_index
########################################################################################################################



###################################### Define Creat opt.json function ##################################################
def Creat_opt_json( current_path, cluster, Min_Multi, cuda_num, offer_pretrain_cluster, Max_Multi, Min_M_error_index, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi ):
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi)  :
          Creat_M_opt_json='{}/DNN_TL_{}/nn_fitting/{}-Final-M{}_opt.json'.format( current_path, cluster, cluster, Present_Multi )
       else: 
          Creat_M_opt_json='{}/DNN_TL_{}/nn_fitting/{}-M{}_opt.json'.format( current_path, cluster, cluster, Present_Multi )
       creat_M_opt_format_json='{}/DNN_TL_{}/nn_fitting/creat_opt_format.json'.format(current_path, cluster)
       lines_input_1 = ''
       with open(creat_M_opt_format_json, 'r') as fread:
           for line in fread.readlines()[0:8]:  # in this case, let's copy line 0-6
                   lines_input_1 += line

       lines_input_2 = ''
       with open(creat_M_opt_format_json, 'r') as fread:
           for line in fread.readlines()[13:34]:  # in this case, let's copy line 0-6
                   lines_input_2 += line

       lines_input_3 = ''
       with open(creat_M_opt_format_json, 'r') as fread:
           for line in fread.readlines()[35:]:  # in this case, let's copy line 0-6
                   lines_input_3 += line

       with open (Creat_M_opt_json, 'w') as f:
           f.truncate()
       with open (Creat_M_opt_json, 'a+') as f:
           f.write(lines_input_1)
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi)   :
              f.write('    "output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/Final-M{}-opt",\n'.format(current_path, cluster, cluster, Present_Multi))
           else:
              f.write('    "output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/M{}-opt",\n'.format(current_path, cluster, cluster, (Present_Multi)))
           f.write('    "gpu": "cuda{}",\n'.format(cuda_num))
           f.write('    "optimization": {\n')
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi)  :
              f.write('    \t"input_file":"{}/DNN_TL_{}/nn_fitting/OUT-{}-gas-Final-M{}/fil_structs.xyz.0",\n'.format(current_path, cluster, cluster, Present_Multi))
              f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index ))

           else:
              f.write('    \t"input_file":"{}/DNN_TL_{}/nn_fitting/OUT-{}-gas-M{}/fil_structs.xyz.0",\n'.format(current_path, cluster, cluster, Present_Multi))
              f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index ))

           f.write(lines_input_2)
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index ))
           else:
              f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index ))
           f.write(lines_input_3)
####################################################################################################################


######################################## Define Run opt.json function #################################################
def Run_opt_json( current_path, cluster, Min_Multi, Max_Multi, offer_pretrain_cluster, Min_M_error_index, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi ):
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi)  :
          Creat_M_opt_json='{}/DNN_TL_{}/nn_fitting/{}-Final-M{}_opt.json'.format( current_path, cluster, cluster, Present_Multi )
       else:
          Creat_M_opt_json='{}/DNN_TL_{}/nn_fitting/{}-M{}_opt.json'.format( current_path, cluster, cluster, Present_Multi )
       lines_input_1 = ''
       with open(Creat_M_opt_json, 'r') as fread:
           for line in fread.readlines()[0:12]:  # in this case, let's copy line 0-6
                   lines_input_1 += line

       lines_input_2 = ''
       with open(Creat_M_opt_json, 'r') as fread:
           for line in fread.readlines()[13:34]:  # in this case, let's copy line 0-6
                   lines_input_2 += line

       lines_input_3 = ''
       with open(Creat_M_opt_json, 'r') as fread:
           for line in fread.readlines()[35:]:  # in this case, let's copy line 0-6
                   lines_input_3 += line

       with open (Creat_M_opt_json, 'w') as f:
           f.truncate()
       with open (Creat_M_opt_json, 'a+') as f:
            f.write(lines_input_1)
            if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
               f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index ))
            else:
               f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index ))
            f.write(lines_input_2)
            if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
               f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/Final-M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index ))
            else:
               f.write('    \t"load_network": "{}/DNN_TL_{}/nn_fitting/OUT-{}-trained-NN/M{}nn/fit_network.dill.{}",\n'.format(current_path, cluster, cluster, Present_Multi, Min_M_error_index))
            f.write(lines_input_3)
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/Final-M{}-opt".format(current_path, cluster, cluster, Present_Multi)
       else:
          mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/M{}-opt".format(current_path, cluster, cluster, Present_Multi)
       mkdir(mkpath)

       print('*********************************************************************************************')
       for k in range(0,6): 
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              print('********************************* Run {}-Final-M{}_opt.json ************************************'.format(cluster, Present_Multi))
           else:
              print('********************************* Run {}-M{}_opt.json ************************************'.format(cluster, Present_Multi))
       print('*********************************************************************************************')
      
       os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          os.system('acnnmain {}-Final-M{}_opt.json'.format(cluster, Present_Multi))
       else:
          os.system('acnnmain {}-M{}_opt.json'.format(cluster, Present_Multi))

#######################################################################################################################


##################################### Pick lower energies structs after opt ###########################################
def Pick_lower_energies_structs( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, pick_lower_energy_num, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi ):
       os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))

       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          Creat_pick_py='{}/DNN_TL_{}/nn_fitting/pick_Final-M{}_{}_lowest_energy_after_opt.py'.format( current_path, cluster, Present_Multi, pick_lower_energy_num )
       else:
          Creat_pick_py='{}/DNN_TL_{}/nn_fitting/pick_M{}_{}_lowest_energy_after_opt.py'.format( current_path, cluster, Present_Multi, pick_lower_energy_num )
       creat_pick_format_py='{}/DNN_TL_{}/nn_fitting/creat_pick_lowest_energies_format.py'.format(current_path, cluster)
       lines_input_1 = ''
       with open(creat_pick_format_py, 'r') as fread:
           for line in fread.readlines()[0:23]:  # in this case, let's copy line 0-6
                   lines_input_1 += line

       lines_input_2 = ''
       with open(creat_pick_format_py, 'r') as fread:
           for line in fread.readlines()[24:34]:  # in this case, let's copy line 0-6
                   lines_input_2 += line

       lines_input_3 = ''
       with open(creat_pick_format_py, 'r') as fread:
           for line in fread.readlines()[36:]:  # in this case, let's copy line 0-6
                   lines_input_3 += line

       with open (Creat_pick_py, 'w') as f:
           f.truncate()
       with open (Creat_pick_py, 'a+') as f:
           f.write(lines_input_1)
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              f.write("st = Read_xyz('{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/Final-M{}-opt/opt_structs.xyz.0')\n".format(current_path, cluster, cluster, Present_Multi))
           else:
              f.write("st = Read_xyz('{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/M{}-opt/opt_structs.xyz.0')\n".format(current_path, cluster, cluster, Present_Multi))
           f.write(lines_input_2)
           f.write('min300idx = np.argsort(enarr[:,1])[:{}]\n'.format(pick_lower_energy_num))
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              f.write("output_file = new_file_name('{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/Final-M{}-opt/opt_{}_structs.xyz')\n".format(current_path, cluster, cluster, Present_Multi, pick_lower_energy_num))
           else: 
              f.write("output_file = new_file_name('{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/M{}-opt/opt_{}_structs.xyz')\n".format(current_path, cluster, cluster, Present_Multi, pick_lower_energy_num))
           f.write(lines_input_3)

       print('*********************************************************************************************')
       for k in range(0,6):
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              print('********************************* Pick-{}-Final-M{}-lowest-energy-structs ************************************'.format( cluster, Present_Multi ))
           else:
              print('********************************* Pick-{}-M{}-lowest-energy-structs ************************************'.format( cluster, Present_Multi ))
       print('*********************************************************************************************')
       
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          os.system('python pick_Final-M{}_{}_lowest_energy_after_opt.py'.format( Present_Multi, pick_lower_energy_num ))        
       else:
          os.system('python pick_M{}_{}_lowest_energy_after_opt.py'.format( Present_Multi, pick_lower_energy_num ))

########################################################################################################################

########################################### Filter duplication ########################################################
def Filter_duplication(current_path, cluster, Min_Multi, Max_Multi, offer_pretrain_cluster, pick_lower_energy_num, Max_diff_for_filter, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi ):
       os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))

       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-Final-M{}-filter-for-DFT".format(current_path, cluster, cluster, Present_Multi)
       else:
          mkpath = "{}/DNN_TL_{}/nn_fitting/OUT-{}-M{}-filter-for-DFT".format(current_path, cluster, cluster, Present_Multi )
       mkdir(mkpath)
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          Filter_file = '{}/DNN_TL_{}/nn_fitting/{}-Final-M{}-filter-for-DFT.json'.format(current_path, cluster, cluster, Present_Multi)
          Draw_file   = '{}/DNN_TL_{}/nn_fitting/{}-M{}-draw.json'.format(current_path, cluster, cluster, Present_Multi)

       else:
          Filter_file = '{}/DNN_TL_{}/nn_fitting/{}-M{}-filter-for-DFT.json'.format(current_path, cluster, cluster, Present_Multi)
          Draw_file   = '{}/DNN_TL_{}/nn_fitting/{}-M{}-draw.json'.format(current_path, cluster, cluster, Present_Multi)

       with open (Filter_file, 'w') as f:
           f.truncate()
       with open (Filter_file, 'a+') as f:
           f.write('{\n')
           f.write('"tasks": ["filter"],\n')
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              f.write('"output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-Final-M{}-filter-for-DFT",\n'.format(current_path, cluster,  cluster, Present_Multi))
           else:
              f.write('"output_dir": "{}/DNN_TL_{}/nn_fitting/OUT-{}-M{}-filter-for-DFT",\n'.format(current_path, cluster,  cluster, Present_Multi))
           f.write('"random_seed": 0,\n')
           f.write('"filtering": {\n')
           f.write('"sort": true,\n')
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              f.write('"input_file": "{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/Final-M{}-opt/opt_{}_structs.xyz.0",\n'.format(current_path, cluster, cluster, Present_Multi, pick_lower_energy_num ))
           else:
              f.write('"input_file": "{}/DNN_TL_{}/nn_fitting/OUT-{}-nn-opt-result/M{}-opt/opt_{}_structs.xyz.0",\n'.format(current_path, cluster, cluster, Present_Multi, pick_lower_energy_num ))
           f.write('"align": true,\n')
           f.write('"pre_sort": true,\n')
           f.write('"max_diff_report": 1.0,')
           f.write('"max_diff": {}\n'.format(Max_diff_for_filter))
           f.write('\t}\n')
           f.write('}')
       print('*********************************************************************************************')
       for k in range(0,6):
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              print('********************************* Run {}-Final-M{}-filter-for-DFT.json ************************************'.format(cluster, Present_Multi))
           else:
              print('********************************* Run {}-M{}-filter-for-DFT.json ************************************'.format(cluster, Present_Multi))
       print('*********************************************************************************************')
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          os.system('acnnmain {}-Final-M{}-filter-for-DFT.json'.format(cluster, Present_Multi))
       else:
          os.system('acnnmain {}-M{}-filter-for-DFT.json'.format(cluster, Present_Multi ))

########################################################################################################################

############################################### Write Gaussian input ###################################################
def Write_Gaussian_input( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, input_information, DFT_steps, charge, key_word, memory, nprocshared, element, num_each_element, basis_set_info_line, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi ):
   os.chdir('{}/DNN_TL_{}/nn_fitting'.format(current_path, cluster))

   # In[1]:

   import numpy as np
   import pandas as pd
   import sys
   sys.path.append(os.path.dirname(os.path.expanduser('~/Tools/')))
   from read_xyz import Read_xyz
   from write_gjf_auto import write_gjf

   pd.options.display.max_rows = None
   pd.options.display.max_colwidth = 100
   pd.options.display.float_format = '{:.8f}'.format
   # ### Load data from structures
   # In[2]:
   
   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
      mkpath = "{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set  )
      st = Read_xyz('{}/DNN_TL_{}/nn_fitting/OUT-{}-Final-M{}-filter-for-DFT/fil_structs.xyz.0'.format(current_path, cluster,  cluster, Present_Multi))

   else:
      mkpath = "{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set  )
      st = Read_xyz('{}/DNN_TL_{}/nn_fitting/OUT-{}-M{}-filter-for-DFT/fil_structs.xyz.0'.format(current_path, cluster,  cluster, Present_Multi))
   mkdir(mkpath)
   en, xyz = st.get_xyz()
   enarr, xyzarr = st.df2array()
   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :      
      write_gjf(output_path='{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set ),
                 format_file      = input_information,
                 cluster          = cluster,
                 basis_set_info_line  = basis_set_info_line , # in this case, let's copy line 18-end
                 num_atoms        = st.num_atoms,
                 num_frames       = st.num_frames,
                 xc_function      = xc_function,
                 basis_set        = basis_set,
                 DFT_steps        = DFT_steps,
                 charge           = charge,
                 key_word         = key_word,
                 memory           = memory,
                 nprocshared      = nprocshared,
                 coord            = xyz,
                 el               = element,
                 num_each_element = num_each_element,
                 M                = Present_Multi
                 )


   else:
      write_gjf(output_path='{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set ),
                 format_file      = input_information,
                 cluster          = cluster,
                 basis_set_info_line  = basis_set_info_line , # in this case, let's copy line 18-end
                 num_atoms        = st.num_atoms,
                 num_frames       = st.num_frames,
                 xc_function      = xc_function,
                 basis_set        = basis_set,
                 DFT_steps        = DFT_steps,
                 charge           = charge,
                 key_word         = key_word,
                 memory           = memory,
                 nprocshared      = nprocshared,
                 coord            = xyz,
                 el               = element,
                 num_each_element = num_each_element,
                 M                = Present_Multi
                 )
   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
      Draw_file   = '{}/DNN_TL_{}/nn_fitting/{}-Final-M{}-draw.json'.format(current_path, cluster, cluster, Present_Multi)
   else:
      Draw_file   = '{}/DNN_TL_{}/nn_fitting/{}-M{}-draw.json'.format(current_path, cluster, cluster, Present_Multi)

   with open (Draw_file, 'w') as f:
       f.truncate()
   with open (Draw_file, 'a+') as f:
       f.write('{\n')
       f.write('"tasks": ["draw"],\n')
       f.write('"output_dir": "{}",\n'.format( mkpath  ))
       f.write('"random_seed": 0,\n')
       f.write('"report": {\n')
       if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
          f.write('"input_file": "{}/DNN_TL_{}/nn_fitting/OUT-{}-Final-M{}-filter-for-DFT/fil_structs.xyz.0",\n'.format(current_path, cluster,  cluster, Present_Multi ))
       else:
          f.write('"input_file": "{}/DNN_TL_{}/nn_fitting/OUT-{}-M{}-filter-for-DFT/fil_structs.xyz.0",\n'.format(current_path, cluster,  cluster, Present_Multi ))
       f.write('"number": 9999,\n')
       f.write('"ratio": 1.6\n')
       f.write('\t}\n')
       f.write('}')
   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) : 
      os.system('acnnmain {}-Final-M{}-draw.json'.format(cluster, Present_Multi )) 
   else:
      os.system('acnnmain {}-M{}-draw.json'.format(cluster, Present_Multi ))

   if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
      os.rename("{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}/report.pdf".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set) , "{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}/{}-Final-M{}-DNN-opt-structs.pdf".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi ))
   else:
      os.rename("{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}/report.pdf".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set) , "{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}/{}-M{}-DNN-opt-structs.pdf".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi ) )

################################################################################################


############################################ Define Filter_AUTO function ###############################################
def Filter_AUTO(current_path, cluster, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, Max_diff_for_filter, Max_diff_for_filter_change, Filter, pick_lower_energy_num, input_information, DFT_steps, charge, key_word, memory, nprocshared, element, num_each_element, basis_set_info_line, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi ):

      for j in range( 1,int( ( 1-float(Max_diff_for_filter) )/0.09) ):
           if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
              os.chdir('{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set ))
           else:
              os.chdir('{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set ))

           gjf_num = Counter_files()
           if gjf_num > 45 and Max_diff_for_filter_change < 0.9 and Filter == 'ON' :
              if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
                 file_path_1 = "{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set  )
              else:
                 file_path_1 = "{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set  )
              shutil.rmtree(file_path_1)

              if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
                 file_path_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-Final-M{}-filter-for-DFT".format(current_path, cluster, cluster, Present_Multi )
              else:
                 file_path_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-M{}-filter-for-DFT".format(current_path, cluster, cluster, Present_Multi )
              shutil.rmtree(file_path_2)
              Max_diff_for_filter_change = Max_diff_for_filter_change + 0.09
              Max_diff_for_filter        = Max_diff_for_filter_change
              Filter_duplication(current_path, cluster, Min_Multi, Max_Multi, offer_pretrain_cluster, pick_lower_energy_num, Max_diff_for_filter, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )
              Write_Gaussian_input( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, input_information, DFT_steps, charge, key_word, memory, nprocshared, element, num_each_element, basis_set_info_line, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )

           elif gjf_num < 15 and Max_diff_for_filter_change > 0.15 and Filter == 'ON':
              if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
                 file_path_1 = "{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set  )
                 file_path_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-Final-M{}-filter-for-DFT".format(current_path, cluster, cluster, Present_Multi )
              else:
                 file_path_1 = "{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}".format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set  )
                 file_path_2 = "{}/DNN_TL_{}/nn_fitting/OUT-{}-M{}-filter-for-DFT".format(current_path, cluster, cluster, Present_Multi )
              shutil.rmtree(file_path_1)
              shutil.rmtree(file_path_2)
              Max_diff_for_filter_change = Max_diff_for_filter_change - 0.09
              Max_diff_for_filter        = Max_diff_for_filter_change
              Filter_duplication(current_path, cluster, Min_Multi, Max_Multi, offer_pretrain_cluster, pick_lower_energy_num, Max_diff_for_filter, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )
              Write_Gaussian_input( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, input_information, DFT_steps, charge, key_word, memory, nprocshared, element, num_each_element, basis_set_info_line, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )
           
      #     if gjf_num < 35 and Max_diff_for_filter_change < 0.9 and Filter == 'ON' :
      #        break 
########################################################################################################################

############################# 将 .gjf 文件复制到指定文件夹 #####################################
def copy_and_scp(current_path, cluster, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, server_num, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi):
      # srcfile 需要复制、移动的文件   
      # dstpath 目的地址
      import os, sys, stat
      
      if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
         src_dir = '{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}/'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set )

         dst_dir1 = '{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}_server{}/'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, server_num[0] )
    
      else:
         src_dir = '{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}/'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set )

         dst_dir1 = '{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}_server{}/'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, server_num[int( (Present_Multi - Min_Multi)/2 )] )

      mkdir(dst_dir1)

      src_file_list1 =  glob.glob(src_dir + '*.gjf')
      counter_gjf_num = 0
      for srcfile in src_file_list1:
          mycopyfile(srcfile, dst_dir1)
          counter_gjf_num = counter_gjf_num + 1
########################################################################################################################
      if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
         Gussian_run_file1_1 = '{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}_server{}/'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, server_num[0] )
      else:
         Gussian_run_file1_1 = '{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}_server{}/'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set, server_num[int( (Present_Multi - Min_Multi)/2 )] )
      f = open(Gussian_run_file1_1 + 'run', "w")
      with open(Gussian_run_file1_1 + 'run', 'w') as f:
           f.truncate()
      with open (Gussian_run_file1_1 + 'run', 'a+') as f:
           for j in range(1,(counter_gjf_num+1)):
               f.write('g{} {}-M{}-{}-{}-{}.gjf\n'.format(Guassian_tpye, cluster, Present_Multi, xc_function, basis_set, j ))
      if Run_Final_Min_Multi =='ON' and int(Final_Min_Multi) == int(Present_Multi) :
         os.chdir('{}/DNN_TL_{}/nn_fitting/DFT_{}_Final-M{}_{}_{}/Final_DFT_{}_Final-M{}_{}_{}'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set ))
         os.system('scp -r Final_DFT_{}_Final-M{}_{}_{}_server{} {}@server{}:/home/{}'.format(cluster, Present_Multi, xc_function, basis_set, server_num[0], current_path, server_num[0], user_name))
      else:
         os.chdir('{}/DNN_TL_{}/nn_fitting/DFT_{}_M{}_{}_{}/Final_DFT_{}_M{}_{}_{}'.format(current_path, cluster, cluster, Present_Multi, xc_function, basis_set, cluster, Present_Multi, xc_function, basis_set ))
         os.system('scp -r Final_DFT_{}_M{}_{}_{}_server{} {}@server{}:/home/{}'.format(cluster, Present_Multi, xc_function, basis_set, server_num[int( (Present_Multi - Min_Multi)/2 )], current_path, server_num[int( (Present_Multi - Min_Multi)/2 )], user_name))
########################################################################################################################


#********************************************* Train DNN network *******************************************************************************************************************************************************************************
if Train_DNN == 'ON':
   Min_Pretrain_nn_error_index = 0
   Min_Refnn_error_index       = 0
   if Transfer_learning_diff == 'YES':

#************************************** Creat get_pretrain_network.json ************************************************
      Creat_get_pretrain_network_json(current_path, cluster, offer_pretrain_cluster, pretrain_cluster_Min_Multi, cuda_num, Transfer_learning_diff, Min_Multi )

#*************************************** Run get_pretrain_network.json *************************************************
      Min_Pretrain_nn_error_index = Run_get_pretrain_network_json( current_path, cluster ) 

#****************************************** Creat Refnn_network.json ***************************************************
   if cluster != offer_pretrain_cluster  or (cluster == offer_pretrain_cluster and Min_Multi == pretrain_cluster_Min_Multi) :
      Creat_Refnn_network_json( current_path, cluster, Min_Multi, cuda_num, offer_pretrain_cluster, pretrain_cluster_Min_Multi, Min_Pretrain_nn_error_index, Refnn_epochs, Transfer_learning_diff )

#******************************************** Run Refnn_network.json ***************************************************
      Min_Refnn_error_index =  Run_Refnn_network_json( current_path, cluster, Min_Multi )

########################################################################################################################
 
#*************************************** Run Other_network.json and opt.json *******************************************
Min_M_error_index = 0

list_Multi = []

if cluster != offer_pretrain_cluster  or (cluster == offer_pretrain_cluster and Min_Multi == pretrain_cluster_Min_Multi) :
    for i in range( 1, int(0.5 * (Max_Multi-Min_Multi) + 1)):
        Multi = Min_Multi + 2 * i
        list_Multi.append(Multi)
else:
    for i in range( 0, int(0.5 * (Max_Multi-Min_Multi) + 1)):
        Multi = Min_Multi + 2 * i
        list_Multi.append(Multi)

if Run_Final_Min_Multi == 'ON' and Min_Multi != Max_Multi :
   list_Multi.append(Final_Min_Multi) 

print('list_Multi:',list_Multi)
for Present_Multi in list_Multi:

    if Train_DNN == 'ON':

       Creat_Other_networks_json( Max_Multi, Min_Multi, current_path, cluster, cuda_num, offer_pretrain_cluster, pretrain_cluster_Min_Multi, other_Multi_epochs, Min_Refnn_error_index, Min_M_error_index, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi, Transfer_from_Refnn, Refnn_Multi, Multi_Transfer_to_Final )
       Min_M_error_index = Run_Other_network_json( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi, Transfer_from_Refnn, Refnn_Multi )     

    if Creat_initial_structs == 'ON':
       Creat_and_Run_creat_json( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, cluster_name, num_initial_structs, d_length, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )

    if Use_DNN_to_opt_structs == 'ON':
       if Train_DNN == 'OFF':
          Min_M_error_index = 0
       Creat_opt_json( current_path, cluster, Min_Multi, cuda_num, offer_pretrain_cluster, Max_Multi, Min_M_error_index, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi)
       Run_opt_json( current_path, cluster, Min_Multi, Max_Multi, offer_pretrain_cluster, Min_M_error_index, Present_Multi,Final_Min_Multi, Run_Final_Min_Multi )

    if Pick_structs == 'ON':
       Pick_lower_energies_structs( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, pick_lower_energy_num, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi ) 

    if Filter == 'ON':
       Filter_duplication(current_path, cluster, Min_Multi, Max_Multi, offer_pretrain_cluster, pick_lower_energy_num, Max_diff_for_filter, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )

    if Write_Full_steps_Gaussian_input == 'ON':
       Max_diff_for_filter_change = float(Max_diff_for_filter)
       Write_Gaussian_input( current_path, cluster, Max_Multi, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, input_information, DFT_steps, charge, key_word, memory, nprocshared, element, num_each_element, basis_set_info_line, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )
       Filter_AUTO(current_path, cluster, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, Max_diff_for_filter, Max_diff_for_filter_change, Filter, pick_lower_energy_num, input_information, DFT_steps, charge, key_word, memory, nprocshared, element, num_each_element, basis_set_info_line, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi )
       copy_and_scp(current_path, cluster, Min_Multi, offer_pretrain_cluster, xc_function, basis_set, server_num, Present_Multi, Final_Min_Multi, Run_Final_Min_Multi)

#######################################################################################################################






