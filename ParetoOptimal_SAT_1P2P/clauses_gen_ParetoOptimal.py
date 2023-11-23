import numpy as np
import pandas as pd
from itertools import combinations,combinations_with_replacement

import os, sys,copy, math, re, time, timeit

from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from functions import*
from Smart_Pair import*
import subprocess


TC = Time_Counter_v2()
stage1_Total_time_counter_start = time.perf_counter()

###### Input vars
### input
# @input vars
data_file_path = sys.argv[1]
tree_depth_input = int(sys.argv[2])
epsilon_input = float(sys.argv[3])
consts_path = sys.argv[4]
tmp_solution_path = sys.argv[5]
## smart pair
global_solver_time_limit = int(sys.argv[6])


stage1MsgOutPath = tmp_solution_path+"phase_1_out_print.txt"
try:
    os.remove(stage1MsgOutPath)
except OSError:
    pass
stage1MsgOut = open(stage1MsgOutPath, "a")

old_stdout = sys.stdout
sys.stdout = stage1MsgOut

###### Data
# consts
ML_base,CL_base = get_ML_CL(consts_path)
# df
data_file_path = "content/" + data_file_path

# get the data frame and the specifications of the df
X,y,label_pos, num_rows, num_features, num_labels = get_df_and_spec(data_file_path)
# normalization on X
df = pd.DataFrame(MinMaxScaler().fit_transform(X) * 100,
                  index=X.index,
                  columns=X.columns)

## Distance Class
# Get distance class if drop the first DC, [1:]
df_np = np.array(df)
Distance_Class = get_distance_class(df_np, epsilon_input)

with open(tmp_solution_path + 'DC', 'w')as DCFile:
	for i in Distance_Class:
		tmp_w_string = '-'.join(map(str, i))
		DCFile.write(tmp_w_string + "\n")

##### Tree vars
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
      f"num label: {n_labels}\n"
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
start_ind +=  len(Distance_Class)

print(f'a: {a[0,0]} - {a[-1,-1]} -> {a[-1,-1] - a[0,0] +1}  \n'
      f's: {s[0,0]} - {s[-1,-1]} -> {s[-1,-1] - s[0,0] +1}\n'
      f'z: {z[0,0]} - {z[-1,-1]} -> {z[-1,-1] - z[0,0] +1}\n'
      f'g: {g[0,0]} - {g[-1,-1]} -> {g[-1,-1] - g[0,0] +1}\n'
      f'x: {x[0,0]} - {x[-1,-1]} -> {x[-1,-1] - x[0,0] +1}\n'
      f'b0: {b0[0]} - {b0[-1]} -> {b0[-1] - b0[0] +1}\n'
      f'b1: {b1[0]} - {b1[-1]} -> {b1[-1] - b1[0] +1}')

print('variables***')

# ************** Vars
# ************** Vars
NUM_DISTANCE_CLASS = len(Distance_Class)
# 2*NUM_DISTANCE_CLASS for md+ms
HARD_CLAUSE_W = 2*NUM_DISTANCE_CLASS +1
SOFT_CLAUSE_W = 1
Num_VARS = start_ind-1

######################## Clause Gen ###############################

# (7)
# (!a_t,j, !a_t,j')
tmp_time_counter_start = time.perf_counter()

feature_comb_ind = np.array(list(combinations(feature_index, 2)))
clause_list_7 = np.dstack((-a[:, feature_comb_ind[:, 0]], -a[:, feature_comb_ind[:, 1]])).reshape(-1, 2)

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '7',
               clause_list_7.shape )
## No need to add And(c), as all clauses are hard clauses

# (8)
# this is just the rows of a are the clauses
tmp_time_counter_start = time.perf_counter()
clause_list_8 = a

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '8',
               clause_list_8.shape )
# (9)
tmp_time_counter_start = time.perf_counter()

clause_list_9 = np.array([]).reshape(0, 3)
for j in range(n_feature):
    # j=[0,1,2,3][j%4]
    # t=np.arange(n_bnodes)[j%len(np.arange(n_bnodes))]
    # for t in range(n_bnodes):
    sort_ind = get_sorted_index(df, df.columns[j])
    ind = np.repeat(sort_ind, 2)[1:-1].reshape(-1, 2)
    # write a each chunk to file by 'j'
    clause_list_9 = np.vstack((clause_list_9,
                               np.dstack((
                                         np.repeat(-a[:, j], len(ind), axis=0).reshape(-1, len(ind)), s[ind[:, 0], :].T,
                                         -s[ind[:, 1], :].T)).reshape(-1, 3)
                               ))

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '9',
               clause_list_9.shape )

## (10) (!a_t,j, !s_i,t, s_i',t)
# (10)
tmp_time_counter_start = time.perf_counter()

clause_list_10 = np.array([]).reshape(0, 3)
for b in range(n_bnodes):
    # iterate over features
    for ind_j, name_j in enumerate(df.columns):
        tmp_sort_by_f = df[name_j].sort_values()
        tmp_sorted_consec_pairs = np.repeat(tmp_sort_by_f.index.tolist(), 2)[1:-1].reshape(-1, 2)
        # The index of the same point of the two column is the index of the consecutive pairs
        tmp_eq_pair_ind_list = \
        np.where(tmp_sort_by_f.iloc[:-1].reset_index(drop=True) == tmp_sort_by_f.iloc[1:].reset_index(drop=True))[0]
        if len(tmp_eq_pair_ind_list) > 0:
            clause_list_10 = np.vstack((clause_list_10,
                                        np.vstack((np.repeat(-a[b, ind_j], len(tmp_eq_pair_ind_list)),
                                                   -s[tmp_sorted_consec_pairs[tmp_eq_pair_ind_list, 0], b],
                                                   s[tmp_sorted_consec_pairs[tmp_eq_pair_ind_list, 1], b])).T))

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '10',
               clause_list_10.shape )
## (11) $(\neg z_{i,t}, s_{i,t'}) \quad t \in T_L, x_i \in X, t'\in A_l(t)$
## (12) $(\neg z_{i,t}, \neg s_{i,t'}) \quad t \in T_L, x_i \in X, t'\in A_r(t)$

# (11) (12)
tmp_time_counter_start = time.perf_counter()
clause_list_11 = np.array([]).reshape(0, 2)
clause_list_12 = np.array([]).reshape(0, 2)
# (!z_i,t, s_i,t')
for l in range(n_lnodes):
    _, left_ancestors_list, right_ancestors_list = get_ancestor_nodes(tree_depth, l)

    # 11
    clause_list_11 = np.vstack((clause_list_11, np.hstack(
        (np.repeat(-z[:, l], len(left_ancestors_list)).reshape(-1, 1), s[:, left_ancestors_list].reshape(-1, 1)))))
    # 12
    clause_list_12 = np.vstack((clause_list_12, np.hstack(
        (np.repeat(-z[:, l], len(right_ancestors_list)).reshape(-1, 1), -s[:, right_ancestors_list].reshape(-1, 1)))))

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '11_12',
               [clause_list_11.shape[0]*2,clause_list_11.shape[1]] )

## (13) $(z_{i,t},\underset{t' \in A_l(t)}{\vee} \neg s_{i,t'}, \underset{t' \in A_r(t)}{\vee} s_{i,t'}) \quad t\in T_L, x_i\in X$
# (13)
### For tree depth d, sum of left ancestor and right ancestor equals d.
### Then the length of the clause is the z + s[l&r clauses] -> 1+d
tmp_time_counter_start = time.perf_counter()

clause_list_13 = np.array([]).reshape(0, tree_depth + 1)
for leaf in range(n_lnodes):
    _, left_ancestors_list, right_ancestors_list = get_ancestor_nodes(tree_depth, leaf)

    clause_list_13 = np.vstack((clause_list_13, np.hstack(
        (z[:, leaf].reshape(-1, 1), -s[:, left_ancestors_list], s[:, right_ancestors_list]))))

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '13',
               clause_list_13.shape)
# (14), (15)
# (!a_t,j, s_#(1,j),t)
# (!a_t,j, s_#(-1,j),t)
tmp_time_counter_start = time.perf_counter()

clause_list_14 = np.array([]).reshape(0, 2)
clause_list_15 = np.array([]).reshape(0, 2)
for j in range(n_feature):
    # get the index of dataset sorted on j ascending
    sort_ind = get_sorted_index(df, df.columns[j])

    clause_list_14 = np.vstack((clause_list_14, np.hstack((-a[:, j].reshape(-1, 1), s[sort_ind[0], :].reshape(-1, 1)))))
    clause_list_15 = np.vstack(
        (clause_list_15, np.hstack((-a[:, j].reshape(-1, 1), -s[sort_ind[-1], :].reshape(-1, 1)))))

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '14_15',
               [clause_list_14.shape[0]*2, clause_list_14.shape[1]])

### Extended Hard Clauses
# (16)
# note: ! if num labels < 3, then c is 0 or negative !
# The number of rows equals to the number of leaf node
tmp_time_counter_start = time.perf_counter()

clause_list_16 = np.hstack((g[:, :-2].reshape(-1,1), -g[:, 1:-1].reshape(-1,1)))

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '16',
               clause_list_16.shape)


# (17) (18)
tmp_time_counter_start = time.perf_counter()
clause_list_17, clause_list_18 = [], []  # np.array([]).reshape(0,3),np.array([]).reshape(0,3)
for k in range(n_labels - 1):
    for i in range(n_points):
        for l in range(n_lnodes):
            # (17)
            clause_list_17 += [[-z[i, l], -g[l, k], x[i, k]]]
            # (18)
            clause_list_18 += [[-z[i, l], g[l, k], -x[i, k]]]

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '17_18',
               [len(clause_list_17)*2, len(clause_list_17[0])])

# (19)
# This assumes that the number of data points is at least the same as the
#  number of clusters
# num(n_labels-1) diagnols of x
# num of rows is also num(n_labels-1)
tmp_time_counter_start = time.perf_counter()

clause_list_19 = -x.diagonal()[:n_labels - 1].reshape(-1, 1)

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '19',
               clause_list_19.shape)
# (20)
# use the default ascending index order of the dataset
# Assume number of points larger than number of clusters
# Note: c<i and c starts from 2 (index 1), so i starts from 3 (index 2)
tmp_time_counter_start = time.perf_counter()

clause_list_len_20 = []
clause_list_20=[]
# iterate over each point
for i in range(2, n_points):
    # iterate over c, c<i
    # if i smaller than the number of labels, use the number of labels instead
    # clause_list_20.append(np.vstack((x_[i,1:min(i,n_labels-1) ], x_[:i, 1:min(i,n_labels-1)])).T)
  curr_c_value = min(i, n_labels - 1)
  clause_20_tmp = np.vstack((-x[i, 1:curr_c_value], x[:i, 0:curr_c_value - 1])).T
  # print(
  #   f'--{np.repeat(HARD_CLAUSE_W, n_labels - 2).reshape(-1, 1).shape} - {clause_20_tmp.shape} - {np.zeros(n_labels - 2).reshape(-1, 1)}')
  clause_list_len_20.append(clause_20_tmp.shape)
  clause_list_20.append(clause_20_tmp)

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '20',
               clause_list_len_20[-1])
# (21)
# assume |X|>=K, ensue that minimum k' or maximum k clusters
# all assigned cluster non-empty
# k is the number of clusters
tmp_time_counter_start = time.perf_counter()

clause_list_21 = x[:, -2].reshape(1, -1)

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '21',
               clause_list_21.shape)

# (32) (33) (34)
# total list length is 3*n_DC -2
# w>1 -> w starts from the second DC, that is (index) 1
clause_list_32 = np.vstack((-b0[1:], b0[:-1])).T
clause_list_33 = np.vstack((-b1[1:], b1[:-1])).T
clause_list_34 = np.vstack((-b1, b0)).T
# md/b0 ->32  | ms/b1 ->33
clause_list_dict_32_33_34 ={
    "b0":clause_list_32,
    "b1":clause_list_33,
}

clause_list_dict = {
    "clause_list_7": clause_list_7,
    "clause_list_8": clause_list_8,
    "clause_list_9": clause_list_9,
    "clause_list_10": clause_list_10,
    "clause_list_11": clause_list_11,
    "clause_list_12": clause_list_12,
    "clause_list_13": clause_list_13,
    "clause_list_14": clause_list_14,
    "clause_list_15": clause_list_15,
    "clause_list_16": clause_list_16,
    "clause_list_17": clause_list_17,
    "clause_list_18": clause_list_18,
    "clause_list_19": clause_list_19,
    "clause_list_21": clause_list_21
}




writeMLCL(ML_base, CL_base, f"{tmp_solution_path}stage1_optimalConsts") #_{consts_path.split('/')[-1]}

print(f"--- len ml cl ---   ml: {len(ML_base)}, cl: {len(CL_base)}")




stage1MsgOut.close()


loandra_status, b0_res, b1_res = 'OPTIMUM FOUND',-1,-1
global_solver_time_counter = 0
iter_ind=0
alter_ind_gen = binary_generator() #

while loandra_status=='OPTIMUM FOUND':
    iter_TC = Time_Counter_v2()
    iter_stage2_Total_time_counter_start = time.perf_counter()
    obj_ = ['b0','b1'][next(alter_ind_gen)]
    outfile_handle = open(f'{tmp_solution_path}out_{iter_ind}_{obj_}', 'w')
    sys.stdout=outfile_handle

    curr_b1_sat =np.sum(b1_res)
    curr_b0_sat =np.sum(b0_res)
    print(f'---curr_b1_sat {curr_b1_sat} - curr_b0_sat {curr_b0_sat}')
    # if curr_b1_sat>=len(b1):

    clausesIO_time_counter_start = time.perf_counter()
    # create clauses file
    clause_file_name = tmp_solution_path + 'clauses_' + consts_path.split('/')[-1]
    try:
        os.remove(clause_file_name)
    except OSError:
        pass
    f = open(clause_file_name, "a", 100 * (2 ** 20))  # Textual write and read
    # use ML and CL to enforce b0 b1
    b0b1_MlCl_Consts_path = tmp_solution_path+f"b0b1_MlCl_Consts_{iter_ind}_{obj_}"

    ### Obj  SOFT CLAUSES
    if obj_ == 'b0':    
        np.savetxt(f ,np.hstack((np.repeat(1, len(b0)).reshape(-1,1),
                                    -b0.reshape(-1,1),
                                    np.zeros(len(b0)).reshape(-1,1))), fmt='%d')
        ## b1 (b+) enforcement hard clause
        print(f"************* curr_b1_sat: {curr_b1_sat} #DC: {NUM_DISTANCE_CLASS}| iter_ind_{iter_ind}_{obj_}")
        write_pairs_to_file_withBaseConsts(Distance_Class[:curr_b1_sat+1], filename=b0b1_MlCl_Consts_path, type="ml",
                                            ML_base = ML_base, CL_base = CL_base)
        
    if obj_ == 'b1':
        np.savetxt(f ,np.hstack((np.repeat(1, len(b1)).reshape(-1,1),
                                b1.reshape(-1,1),
                                    np.zeros(len(b1)).reshape(-1,1))), fmt='%d')
        # b0 (b-) enforcement hard clause
        print(f"\n****** curr_b0_sat: {curr_b0_sat} #DC: {NUM_DISTANCE_CLASS}|   - selected dc size: {len(Distance_Class[curr_b0_sat:])}")
        write_pairs_to_file_withBaseConsts(Distance_Class[curr_b0_sat:], filename=b0b1_MlCl_Consts_path, type="cl",
                                           ML_base = ML_base, CL_base = CL_base)
        #
        print(f'[debug] ---------- > len equal {curr_b0_sat==len(b0)} {-b0[min(curr_b0_sat, len(b0)-1)]}  -curb0: {curr_b0_sat} len-1: {len(b0)-1}')
    #
    print(f' ---> [debug] ---> {np.vstack((-b0, b1)).reshape(-1,1).shape}   ndc: {n_DC}')

    ### CLAUSES IO: write clauses 7-26 to file  ###
    for ind, (clause_name, clause_list) in enumerate(clause_list_dict.items()):
        write_clauses_to_file(f, clause_list, HARD_CLAUSE_W)
        print(f"stage2 IO |{clause_name} -size: {len(clause_list)}")
    # write clauses 32 or 33 to file based on obj type
    write_clauses_to_file(f, clause_list_dict_32_33_34[obj_], HARD_CLAUSE_W)
    # write clause 20 to file
    counter=0
    for sub_clause_list in clause_list_20:
        write_clauses_to_file(f, sub_clause_list, HARD_CLAUSE_W)
        counter+=sub_clause_list.shape[0]
    print(f"stage2 IO |clauses20 -size:{counter}")

    f.close()
    clausesIO_time_counter_end = time.perf_counter()
    iter_TC.counter(clausesIO_time_counter_start,
                    clausesIO_time_counter_end,
                    'clausesIO_7-21_32-34',
                    np.nan)

    SmartPair_time_counter_start = time.perf_counter()
    print(data_file_path,consts_path, tmp_solution_path,clause_file_name)
    command = ['java',
                    f'-DconstsPath={b0b1_MlCl_Consts_path}',
                    f'-DdataPath={data_file_path}', 
                    f'-DxStartInd={x[0,0]}', 
                    f'-Db0StartInd={b0[0]}', 
                    f'-DhardClauseWeight={HARD_CLAUSE_W}', 
                    f'-DdistanceClassPath={tmp_solution_path}DC', 
                    f'-DoutFileName={clause_file_name}', 
                    f'-Dobj_='+{"b0":"md", "b1":"ms"}[obj_], 
                    '-jar', 
                    'smart-pair_allObj_j11.jar'] # smart-pair_allObj_j11.jar | smart-pair.jar
    
    print("Smart Pair jar \n"+' '.join(command))
    # os.system(' '.join(command))
    output = subprocess.check_output(command, cwd=os.getcwd())
    output_str = output.decode()
    print(output_str)

    # time counter
    SmartPair_time_counter_end = time.perf_counter()
    iter_TC.counter(SmartPair_time_counter_start,
                    SmartPair_time_counter_end,
                    'smtPr_22-31',
                    np.nan)
    
    # As we using the ML and CL as b+ and b- enforecement, we may reach infeasible solution
    # in smart pair. So we check the output of the smartpair.
    if "infeasible solution -inner pair" in output_str:
        iter_TC.print_dict()
        print("*Infeasible solution detected in SmartPair*")
        print("*loandra status: UNSATISFIABLE\nb0_final: 0 | 0_ind: 0\nb1_final: 0 | 0_ind: 0")
        break


    ### Add header to clause
    tmp_time_counter_start = time.perf_counter()

    Final_Clauses_File_Name = tmp_solution_path + f'phase_2_clauses_final_iter{iter_ind}_{obj_}'
    with open(Final_Clauses_File_Name, 'w') as f0:
        with open(clause_file_name, 'r+') as f1:
            clauses_list_fr_file = f1.readlines()
            num_clauses = len(clauses_list_fr_file)
            loandra_param = f'p wcnf {Num_VARS} {num_clauses} {HARD_CLAUSE_W}\n'
            f0.write(loandra_param)
            for clause in clauses_list_fr_file:
                f0.write(clause)
    print(f'\nloandra header: {loandra_param}')
    # delete the file without handler
    os.remove(clause_file_name)

    tmp_time_counter_end = time.perf_counter()
    iter_TC.counter(tmp_time_counter_start,
            tmp_time_counter_end,
            'cl_file_header',
            np.nan)


    ### call loandra solver
    loandra_res_file_name = tmp_solution_path + f'phase_2_loandra_res_iter{iter_ind}_{obj_}'
    loandra_cmd = f'timeout {global_solver_time_limit-global_solver_time_counter}s ./loandra-master/loandra_static -pmreslin-cglim=30 -weight-strategy=1 -print-model -verbosity=1 ' \
                + Final_Clauses_File_Name+' >' + loandra_res_file_name

    solver_time_counter_start = time.perf_counter()

    os.system(loandra_cmd)

    solver_time_counter_end = time.perf_counter()
    iter_TC.counter(tmp_time_counter_start,
            tmp_time_counter_end,
            '*solver*',
            np.nan)
    
    # calcualte the globle time consumption, added the solver time in this round to the total solver time counter 
    curr_solver_time = solver_time_counter_end-solver_time_counter_start
    global_solver_time_counter+= curr_solver_time
    print(f'Global Solver Time: {global_solver_time_counter} - curr_slvr_time: {curr_solver_time} -sum: {global_solver_time_counter+curr_solver_time}  timelimit: {global_solver_time_limit}')


    iter_stage2_Total_time_counter_end = time.perf_counter()
    iter_TC.counter(iter_stage2_Total_time_counter_start,
            iter_stage2_Total_time_counter_end,
            'Total',
            np.nan)
    # print stat
    iter_TC.print_dict()


    loandra_status, b0_res, b1_res = output_final_stage(loandra_res_file_name, b0,b1,x,y,n_points, n_labels)
    
    # if the global time limit is exceeded, break
    if global_solver_time_counter  >= global_solver_time_limit:
        print(f"*PO break Global Solver Time Limit reached | {global_solver_time_counter} >= {global_solver_time_limit}|")
        break
    if obj_=="b1":
        iter_ind+=1
        # if all b0 sat, then it will automatically get UNSAT when optimizing b0 again, as it enforeces b1+=1, conflicting
        # But if all b1 sat, it will get in infinite loop, as [:#sat_b1+1] still the full set, and we don't enforce b0+=1
        # so we detect if all b1 sat when optmizing b1, finish
        if np.sum(b1_res) == NUM_DISTANCE_CLASS:
            print(f"*PO Break all b+ sat b+size: {np.sum(b1_res)}  -  DC size: {NUM_DISTANCE_CLASS}")
            break
    # terminate the output for this round
    outfile_handle.close()

with open(f"{tmp_solution_path}ParetoOptimalInfo", "w")as f:
    f.write(f"Total iteration: {iter_ind}")


