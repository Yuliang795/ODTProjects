import numpy as np
import pandas as pd
from itertools import combinations,combinations_with_replacement

import os, sys,copy, math, re, time, timeit,subprocess

from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from functions import*

from clauses_gen_7_34 import *


TC = Time_Counter_v2()

## overall time counter
OverAll_time_counter_start = time.perf_counter()

###### Input vars
### input
# @input vars
data_file_path = sys.argv[1]
tree_depth_input = int(sys.argv[2])
epsilon_input = float(sys.argv[3])
consts_path = sys.argv[4]
tmp_solution_path = sys.argv[5]
global_solver_timeout = int(sys.argv[6])

stage1MsgOutPath = tmp_solution_path+"verify_out_print.txt"
try:
    os.remove(stage1MsgOutPath)
except OSError:
    pass
stage1MsgOut = open(stage1MsgOutPath, "a")
old_stdout = sys.stdout
sys.stdout = stage1MsgOut


###### Data
# consts
tmp_time_counter_start = time.perf_counter()

ML,CL = get_ML_CL(consts_path)
  
# ML,CL = np.array([], dtype='int').reshape(-1,2), np.array([], dtype='int').reshape(-1,2)

print(f"--- len ml cl ---   ml: {len(ML)}, cl: {len(CL)}")
# df
data_file_path = "content/" + data_file_path

# get the data frame and the specifications of the df
X,y,label_pos, num_rows, num_features, num_labels = get_df_and_spec(data_file_path)
# normalization on X
df = pd.DataFrame(MinMaxScaler().fit_transform(X) * 100,
                  index=X.index,
                  columns=X.columns)

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               'df_consts',
               np.nan )
## Distance Class
tmp_time_counter_start = time.perf_counter()
## Distance Class
# Get distance class if drop the first DC, [1:]
df_np = np.array(df)
Distance_Class = get_distance_class(df_np, epsilon_input)

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               'DC',
               len(Distance_Class))

##### Tree vars
tmp_time_counter_start = time.perf_counter()

# for a complete tree, we can build the tree structure
# with any given tree depth

# number of feature, points
n_feature = df.shape[1]
n_points = df.shape[0]
# number of labels/clusters
n_labels = int(num_labels) # y.unique()
# number of distance class
n_DC = len(Distance_Class)
# tree depth
tree_depth = tree_depth_input
# get the number of branch node and leaf node based on tree depth
n_bnodes = 2**tree_depth-1
n_lnodes = 2**tree_depth
# index the feature
feature_index = np.arange(df.shape[1])
print(f"num branch_node: {n_bnodes}\n"
      f"num leaf_node: {n_lnodes}\n"
      f"num feature: {n_feature}\n"
      f"tree_depth: {tree_depth}\n"
      f"nDC: {n_DC}\n")

# a
start_ind = 1
a = np.arange(start_ind,start_ind + n_bnodes*n_feature).reshape(n_bnodes,n_feature)
start_ind += n_bnodes*n_feature
#s {n_points}*{n_bnodes}
s = np.arange(start_ind, start_ind + n_points*n_bnodes).reshape(n_points,n_bnodes)
start_ind += n_points*n_bnodes
# z {n_points}*{n_lnodes}
z = np.arange(start_ind, start_ind + n_points*n_lnodes).reshape(n_points,n_lnodes)
start_ind +=  n_points*n_lnodes
# g{n_lnodes}*{n_labels}
g = np.arange(start_ind, start_ind + n_lnodes*n_labels).reshape(n_lnodes,n_labels)
start_ind +=  n_lnodes*n_labels
# x {n_points}*{n_labels}
x = np.arange(start_ind, start_ind + n_points*n_labels).reshape(n_points,n_labels)
start_ind +=  n_points*n_labels
# b0 {n_DC}
b0 = np.arange(start_ind, start_ind + len(Distance_Class))
start_ind +=  len(Distance_Class)
# b1 {n_DC}
b1 = np.arange(start_ind, start_ind + len(Distance_Class))

print(f'a: {a[0,0]} - {a[-1,-1]} -> {a[-1,-1] - a[0,0] +1}  \n'
      f's: {s[0,0]} - {s[-1,-1]} -> {s[-1,-1] - s[0,0] +1}\n'
      f'z: {z[0,0]} - {z[-1,-1]} -> {z[-1,-1] - z[0,0] +1}\n'
      f'g: {g[0,0]} - {g[-1,-1]} -> {g[-1,-1] - g[0,0] +1}\n'
      f'x: {x[0,0]} - {x[-1,-1]} -> {x[-1,-1] - x[0,0] +1}\n'
      f'b0: {b0[0]} - {b0[-1]} -> {b0[-1] - b0[0] +1}\n'
      f'b1: {b1[0]} - {b1[-1]} -> {b1[-1] - b1[0] +1}')

## clause gen parameters


variables_dict = {
    'a': a,
    's': s,
    'z': z,
    'g': g,
    'x': x,
    'b0': b0,
    'b1': b1,
    'n_feature': n_feature,
    'n_points': n_points,
    'n_labels': n_labels,
    'n_DC': n_DC,
    'tree_depth': tree_depth,
    'n_bnodes': n_bnodes,
    'n_lnodes': n_lnodes,
    'feature_index': feature_index,
    'TC': TC,
    'Distance_Class': Distance_Class,
    'df': df,
    'ML': ML,
    'CL': CL
}

# ************** Vars
Num_VARS = b1[-1]

NUM_DISTANCE_CLASS = len(Distance_Class)

# # 2*NUM_DISTANCE_CLASS for md+ms
# HARD_CLAUSE_W = 2*NUM_DISTANCE_CLASS +1
# SOFT_CLAUSE_W = 1

stage1MsgOut.close()
########## Iteration until condition satisfied

SOFT_CLAUSE_W = 1
HARD_CLAUSE_W = 2*NUM_DISTANCE_CLASS + 1

# change the weights for hard clauses
variables_dict['HARD_CLAUSE_W'] = HARD_CLAUSE_W
variables_dict['use_SmartPair'] = True

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               'tree_var',
               np.nan)


######################## Clause Gen ###############################
loandra_status, b0_res, b1_res = 0,0,0
global_time_limit = 

# 'b0', 'b1'
for obj_ in ['b1']:
    outfile_handle = open(f'{tmp_solution_path}out_verify_{obj_}', 'w')
    sys.stdout=outfile_handle

    ### Start file I/O
    clause_file_name = 'clauses_' + consts_path.split('/')[-1]
    try:
        os.remove(clause_file_name)
    except OSError:
        pass
    f = open(clause_file_name, "a",100*(2**20))  # Textual write and read



    ### Obj  SOFT CLAUSES
    # The length of the list of the two objectives is 2*n_DC
    if obj_ == 'b0':    
        np.savetxt(f ,np.hstack((np.repeat(1, len(b0)).reshape(-1,1),
                                  -b0.reshape(-1,1),
                                    np.zeros(len(b0)).reshape(-1,1))), fmt='%d')
    if obj_ == 'b1':
        np.savetxt(f ,np.hstack((np.repeat(1, len(b1)).reshape(-1,1),
                                b1.reshape(-1,1),
                                    np.zeros(len(b1)).reshape(-1,1))), fmt='%d')
    if obj_ == 'msmd':
        np.savetxt(f ,np.hstack((np.repeat(1,2*n_DC).reshape(-1,1), np.vstack((-b0, b1)).reshape(-1,1) , np.zeros(2*n_DC).reshape(-1,1))), fmt='%d')

    print(f' ---> [debug] ---> {np.vstack((-b0, b1)).reshape(-1,1).shape}   ndc: {n_DC}')

    ### Base Hard Clauses
    clause_gen_7_21__32_34(f,**variables_dict)

    ### Finish File I/O
    f.close()

    SmartPair_time_counter_start = time.perf_counter()
    #
    print(data_file_path,consts_path, tmp_solution_path,clause_file_name)
    command = ['java',
                    f'-DdataPath={data_file_path}', 
                    f'-DxStartInd={x[0,0]}', 
                    f'-Db0StartInd={b0[0]}', 
                    f'-DhardClauseWeight={HARD_CLAUSE_W}', 
                    f'-DdistanceClassPath={tmp_solution_path}DC', 
                    f'-DoutFileName={clause_file_name}', 
                    '-jar', 
                    'smart-pair.jar']
    if 'mc0.0' not in consts_path.split('/')[-1]:
        paramConsts = f'-DconstsPath={consts_path}'
        command.insert(1, paramConsts)

    print("Smart Pair jar \n"+' '.join(command))
    # os.system(' '.join(command))
    output = subprocess.check_output(command, cwd=os.getcwd())
    print(output.decode())

    SmartPair_time_counter_end = time.perf_counter()
    TC.counter(SmartPair_time_counter_start,
                                SmartPair_time_counter_end,
                                'smtPr_22-31',
                                np.nan)


    #

    ### Add header to clause
    tmp_time_counter_start = time.perf_counter()

    Final_Clauses_File_Name = tmp_solution_path + f'phase_2_clauses_final_verify_{obj_}'
    with open(Final_Clauses_File_Name, 'w') as f0:
        with open(clause_file_name, 'r+') as f1:
            clauses_list_fr_file = f1.readlines()
            num_clauses = len(clauses_list_fr_file)
            loandra_param = f'p wcnf {Num_VARS} {num_clauses} {HARD_CLAUSE_W}\n'
            f0.write(loandra_param)
            for clause in clauses_list_fr_file:
                f0.write(clause)
    # delete the file without handler
    os.remove(clause_file_name)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                tmp_time_counter_end,
                'cl_file_header',
                np.nan)



    # loandra solver
    loandra_res_file_name = tmp_solution_path + f'phase_2_loandra_res_verify_{obj_}'  #f'{data_file_path.split("_")[-1]}_loandra_res'
    loandra_cmd = f'timeout {global_time_limit}s ~/SAT_Project/loandra-master/loandra_static -pmreslin-cglim=30 -weight-strategy=1 -print-model -verbosity=1 ' \
                + Final_Clauses_File_Name+'>' + loandra_res_file_name

    tmp_time_counter_start = time.perf_counter()
    os.system(loandra_cmd)
    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                tmp_time_counter_end,
                '*solver*',
                np.nan)
    
    OverAll_time_counter_end = time.perf_counter()
    TC.counter(OverAll_time_counter_start,
                OverAll_time_counter_end,
                'Total',
                np.nan )
    # print stat
    TC.print_dict()

    print(f'\nloandra header: {loandra_param}')


    loandra_status, b0_res, b1_res = output_final_stage(loandra_res_file_name, b0,b1,x,y,n_points, n_labels)

    outfile_handle.close()



