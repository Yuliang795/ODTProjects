import numpy as np
import pandas as pd
from itertools import combinations,combinations_with_replacement

import os, sys,copy, math, re, time, timeit

from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import subprocess
from functions import*

TC = Time_Counter_v2()

## overall time counter
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
use_SmartPair = sys.argv[6]
stage1_timeout = sys.argv[7]
# objective
obj_ = sys.argv[8]

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
tmp_time_counter_start = time.perf_counter()

if 'mc0.0' in consts_path.split('/')[-1]:
  ML,CL = np.array([], dtype='int').reshape(-1,2), np.array([], dtype='int').reshape(-1,2)
else:
  ML,CL = get_ML_CL(consts_path)
# ML,CL = np.array([], dtype='int').reshape(-1,2), np.array([], dtype='int').reshape(-1,2)

print(f"--- len ml cl ---   ml: {len(ML)}, cl: {len(CL)}")
# df
data_file_path = "./content/" + data_file_path

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
with open(tmp_solution_path + 'DC', 'w')as DCFile:
	for i in Distance_Class:
		tmp_w_string = '-'.join(map(str, i))
		
		DCFile.write(tmp_w_string + "\n")

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

# ************** Vars
NUM_DISTANCE_CLASS = len(Distance_Class)
# 2*NUM_DISTANCE_CLASS for md+ms
HARD_CLAUSE_W = 2*NUM_DISTANCE_CLASS +1
SOFT_CLAUSE_W = 1
Num_VARS = b1[-1]


tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               'tree_var',
               np.nan)


######################## Clause Gen ###############################

### Start file I/O
clause_file_name = tmp_solution_path + 'clauses_' + consts_path.split('/')[-1]
try:
    os.remove(clause_file_name)
except OSError:
    pass
f = open(clause_file_name, "a",100*(2**20))  # Textual write and read

### Obj  SOFT CLAUSES
#   # (37) Not(b0[w]) weight='1'
#   # (38) b1[w] weight='1'
# The length of the list of the two objectives is 2*n_DC
np.savetxt(f ,np.hstack((np.repeat(SOFT_CLAUSE_W,2*n_DC).reshape(-1,1), np.vstack((-b0, b1)).reshape(-1,1) , np.zeros(2*n_DC).reshape(-1,1))), fmt='%d')

print(f' ---> [debug] ---> {np.vstack((-b0, b1)).reshape(-1,1).shape}   ndc: {n_DC}')
### Base Hard Clauses

# (7)
# (!a_t,j, !a_t,j')
tmp_time_counter_start = time.perf_counter()

feature_comb_ind = np.array(list(combinations(feature_index, 2)))
clause_list_7 = np.dstack((-a[:, feature_comb_ind[:, 0]], -a[:, feature_comb_ind[:, 1]])).reshape(-1, 2)
# write to file
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_7)).reshape(-1, 1), clause_list_7,
                         np.zeros(len(clause_list_7)).reshape(-1, 1))), fmt='%d')

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
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(a)).reshape(-1, 1), a, np.zeros(len(a)).reshape(-1, 1))),
           fmt='%d')
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
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_9)).reshape(-1, 1),
                         clause_list_9,
                         np.zeros(len(clause_list_9)).reshape(-1, 1))), fmt='%d')

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
# write to file
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_10)).reshape(-1, 1), clause_list_10,
                         np.zeros(len(clause_list_10)).reshape(-1, 1))), fmt='%d')

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
# add all the clauses to solver
# 11
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_11)).reshape(-1, 1), clause_list_11,
                         np.zeros(len(clause_list_11)).reshape(-1, 1))), fmt='%d')
# 12
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_12)).reshape(-1, 1), clause_list_12,
                         np.zeros(len(clause_list_12)).reshape(-1, 1))), fmt='%d')
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

# write all the clauses to file
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_13)).reshape(-1, 1), clause_list_13,
                         np.zeros(len(clause_list_13)).reshape(-1, 1))), fmt='%d')

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

# write all the clauses to file
# (14)
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_14)).reshape(-1, 1), clause_list_14,
                         np.zeros(len(clause_list_14)).reshape(-1, 1))), fmt='%d')# (15)
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_15)).reshape(-1, 1), clause_list_15,
                         np.zeros(len(clause_list_15)).reshape(-1, 1))), fmt='%d')

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
# np.savetxt(f, np.repeat(0,5).reshape(1,-1), fmt='%d')
# np.savetxt(f, np.hstack(
#     (np.repeat(HARD_CLAUSE_W, n_lnodes).reshape(-1, 1), clause_list_16, np.zeros(n_lnodes).reshape(-1, 1))), fmt='%d')

write_clauses_to_file(f, clause_list_16, HARD_CLAUSE_W)

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

# add to solver
# (17)
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_17)).reshape(-1, 1), np.array(clause_list_17),
                         np.zeros(len(clause_list_17)).reshape(-1, 1))), fmt='%d')
# (18)
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, len(clause_list_18)).reshape(-1, 1), np.array(clause_list_18),
                         np.zeros(len(clause_list_18)).reshape(-1, 1))), fmt='%d')

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
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, n_labels - 1).reshape(-1, 1),
                         clause_list_19,
                         np.zeros(n_labels - 1).reshape(-1, 1))), fmt='%d')

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
    np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, curr_c_value - 1).reshape(-1, 1),
                             clause_20_tmp,
                             np.zeros(curr_c_value - 1).reshape(-1, 1))), fmt='%d')

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
np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, 1).reshape(1, -1),
                         clause_list_21,
                         np.zeros(1).reshape(1, -1))), fmt='%d')

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '21',
               clause_list_21.shape)




### SMART PAIR clauses
if use_SmartPair!='smart':
    # (22) (23) the cluster in the paper starts from 1, assume index 0
    # k is the number of clusters

    tmp_time_counter_start = time.perf_counter()

    clause_list_22 = np.vstack((x[CL[:, 0], 0], x[CL[:, 1], 0])).T
    write_clauses_to_file(f, clause_list_22, HARD_CLAUSE_W)

    
    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '22',
                        clause_list_22.shape)

    tmp_time_counter_start = time.perf_counter()

    clause_list_23 = np.vstack((-x[CL[:, 0], -2], -x[CL[:, 1], -2])).T
    write_clauses_to_file(f, clause_list_23, HARD_CLAUSE_W)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '23',
                        clause_list_23.shape)
    # (24)
    tmp_time_counter_start = time.perf_counter()

    clause_list_24 = np.hstack((-x[CL[:, 0], :-2].reshape(-1, 1),
                                                            -x[CL[:, 1], :-2].reshape(-1, 1),
                                                            x[CL[:, 0], 1:-1].reshape(-1, 1),
                                                            x[CL[:, 1], 1:-1].reshape(-1, 1)))
    write_clauses_to_file(f, clause_list_24, HARD_CLAUSE_W)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '24',
                        clause_list_24.shape)
    # (25)
    tmp_time_counter_start = time.perf_counter()

    clause_list_25 = np.hstack((-x[ML[:, 0], :-1].reshape(-1, 1), x[ML[:, 1], :-1].reshape(-1, 1)))
    write_clauses_to_file(f, clause_list_25, HARD_CLAUSE_W)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '25',
                        clause_list_25.shape)
    # (26)
    tmp_time_counter_start = time.perf_counter()

    clause_list_26 = np.hstack((x[ML[:, 0], :-1].reshape(-1, 1), -x[ML[:, 1], :-1].reshape(-1, 1)))
    write_clauses_to_file(f, clause_list_26, HARD_CLAUSE_W)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '26',
                        clause_list_26.shape)
    # (27)
    tmp_time_counter_start = time.perf_counter()

    clause_list_27 = []

    for w_ind, p in enumerate(Distance_Class):
        tmp_pair = np.array(p)
        clause_list_27.append(np.vstack((np.repeat(b0[w_ind], len(tmp_pair)), x[tmp_pair[:, 0], 0], x[tmp_pair[:, 1], 0])).T)

    clause_list_27 = np.concatenate(clause_list_27, axis=0)
    write_clauses_to_file(f, clause_list_27, HARD_CLAUSE_W)

    # np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, clause_list_27.shape[0]).reshape(-1, 1),
    #                           clause_list_27,
    #                           np.zeros(clause_list_27.shape[0]).reshape(-1, 1))), fmt='%d')

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '27',
                        [len(clause_list_27), len(clause_list_27[0])])
    # (28)
    tmp_time_counter_start = time.perf_counter()

    clause_list_28 = []

    for w_ind, p in enumerate(Distance_Class):
        tmp_pair = np.array(p)
        clause_list_28.append(
            np.vstack((np.repeat(b0[w_ind], len(tmp_pair)), -x[tmp_pair[:, 0], -2], -x[tmp_pair[:, 1], -2])).T)

    clause_list_28 = np.concatenate(clause_list_28, axis=0)
    # write_clauses_to_file(f, clause_list_28, HARD_CLAUSE_W)
    write_clauses_to_file(f, clause_list_28, HARD_CLAUSE_W)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '28',
                        [len(clause_list_28), len(clause_list_28[0])])
    # (29)
    tmp_time_counter_start = time.perf_counter()

    clause_list_29 = []

    for w_ind, p in enumerate(Distance_Class):
        # print(f"{w_ind} -- {p}")
        tmp_pair = np.array(p)
        clause_list_29.append(np.hstack((
            np.repeat(b0[w_ind], len(tmp_pair) * (n_labels - 2)).reshape(-1, 1),
            -x[tmp_pair[:, 0], :-2].reshape(-1, 1),
            -x[tmp_pair[:, 1], :-2].reshape(-1, 1),
            x[tmp_pair[:, 0], 1:-1].reshape(-1, 1),
            x[tmp_pair[:, 1], 1:-1].reshape(-1, 1)))
        )

    clause_list_29 = np.concatenate(clause_list_29, axis=0)
    write_clauses_to_file(f, clause_list_29, HARD_CLAUSE_W)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '29',
                        [len(clause_list_29), 0 if not len(clause_list_29) else len(clause_list_29[0])])

    # (30) (31)
    tmp_time_counter_start = time.perf_counter()

    clause_list_30, clause_list_31 = [], []

    for w_ind, p in enumerate(Distance_Class):
        # print(f"{w_ind} -- {p}")
        tmp_pair = np.array(p)
        # (30)
        clause_list_30.append(np.hstack((np.repeat(-b1[w_ind], len(tmp_pair) * (n_labels - 1)).reshape(-1, 1),
                                                                        -x[tmp_pair[:, 0], :-1].reshape(-1, 1),
                                                                        x[tmp_pair[:, 1], :-1].reshape(-1, 1)
                                                                        ))
                                                    )
        # (31)
        clause_list_31.append(np.hstack((np.repeat(-b1[w_ind], len(tmp_pair) * (n_labels - 1)).reshape(-1, 1),
                                                                        x[tmp_pair[:, 0], :-1].reshape(-1, 1),
                                                                        -x[tmp_pair[:, 1], :-1].reshape(-1, 1)
                                                                        ))
                                                    )

    # (30)
    clause_list_30 = np.concatenate(clause_list_30, axis=0)
    write_clauses_to_file(f, clause_list_30, HARD_CLAUSE_W)

    # (31)
    clause_list_31 = np.concatenate(clause_list_31, axis=0)
    write_clauses_to_file(f, clause_list_31, HARD_CLAUSE_W)

    tmp_time_counter_end = time.perf_counter()
    TC.counter(tmp_time_counter_start,
                        tmp_time_counter_end,
                        '30_31',
                        [len(clause_list_30) * 2, len(clause_list_30[0])])
    # # write the clauses generated in smart pair to file
    #
    tmp_time_counter_start = time.perf_counter()


# (32) (33) (34)
# total list length is 3*n_DC -2
# w>1 -> w starts from the second DC, that is (index) 1
tmp_time_counter_start = time.perf_counter()

clause_list_32_33_34 = np.vstack((np.vstack((-b0[1:], b0[:-1])).T,
                                  np.vstack((-b1[1:], b1[:-1])).T,
                                  np.vstack((-b1, b0)).T))

np.savetxt(f, np.hstack((np.repeat(HARD_CLAUSE_W, 3 * n_DC - 2).reshape(-1, 1),
                         clause_list_32_33_34,
                         np.zeros(3 * n_DC - 2).reshape(-1, 1))), fmt='%d')

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '32_33_34',
               clause_list_32_33_34.shape)
### Finish File I/O
f.close()

if use_SmartPair=='smart':
    SmartPair_time_counter_start = time.perf_counter()
    print(data_file_path,consts_path, tmp_solution_path,clause_file_name)

    command = ['java',
                    f'-DdataPath={data_file_path}', 
                    f'-DxStartInd={x[0,0]}', 
                    f'-Db0StartInd={b0[0]}', 
                    f'-DhardClauseWeight={HARD_CLAUSE_W}', 
                    f'-DdistanceClassPath={tmp_solution_path}DC', 
                    f'-DoutFileName={clause_file_name}', 
                    f'-Dobj_={obj_}', 
                    '-jar', 
                    'smart-pair_obj_j11.jar'] # smart-pair_obj_j11.jar  | smart-pair.jar
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
                'smtPr_22_31',
                np.nan)


#

### Add header to clause
tmp_time_counter_start = time.perf_counter()

Final_Clauses_File_Name = tmp_solution_path + 'phase_1_clauses_final'
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
loandra_res_file_name = tmp_solution_path + 'loandra_res'  #f'{data_file_path.split("_")[-1]}_loandra_res'
loandra_cmd = f'timeout {stage1_timeout}s ./loandra-master/loandra_static -pmreslin-cglim=30 -weight-strategy=1 -print-model -verbosity=1 ' \
              + Final_Clauses_File_Name+'>' + loandra_res_file_name

tmp_time_counter_start = time.perf_counter()

os.system(loandra_cmd)

tmp_time_counter_end = time.perf_counter()
TC.counter(tmp_time_counter_start,
               tmp_time_counter_end,
               '*solver*',
               np.nan)

stage1_Total_time_counter_end = time.perf_counter()
TC.counter(stage1_Total_time_counter_start,
               stage1_Total_time_counter_end,
               'Total',
               np.nan )
# print stat
TC.print_dict()

print(f'\nloandra header: {loandra_param}')

output_final_stage(loandra_res_file_name, b0,b1,x,y,n_points, n_labels)

stage1MsgOut.close()