import numpy as np
import pandas as pd
import math,re, time, timeit,os
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# read and extract the data from file
# param: data file path
# return: X,y,label_pos, num_rows, num_features, num_labels
def get_df_and_spec(data_file_path):
  df = pd.read_csv(data_file_path, header=None, delimiter=r"\s+", skiprows=4)
  with open(data_file_path) as myfile:
    data_spec = [next(myfile).strip("""\n: """).split()[-1] for x in range(4)]
  # data file specifications
  label_pos, num_rows, num_features, num_labels = data_spec
  # label at the end of each line
  label_pos_index = df.shape[1] - 1 if label_pos == "l" else 0
  # seperate variables and the label
  y = df.iloc[:, label_pos_index]
  X = df.drop(label_pos_index, axis=1)
  return X,y,label_pos, num_rows, num_features, num_labels


# split ML and CL consts into parts by percentage
# param: ML, CL cosnts: nparray, percentage (split file by),
# shuffle: whether shuffle the ML,Cl consts
# return: set of parts of ML and CL

def get_ML_CL_parts(ML, CL, fold=3, shuffle_flag=False):
  
  # split consts into parts
  ML_parts_set = np.array_split(ML, fold)
  CL_parts_set = np.array_split(CL, fold)

  return ML_parts_set, CL_parts_set

# this function takes the dataset and selected feature j as the input
# and return the index of dataset sorted bsaed on feature j
def get_sorted_index(dt, j, ascending=True):
    return np.array(dt.sort_values(by=[j], ascending=ascending).index)

# note: leaf node and brach node are developed separately
# both of them starts from index 0.

# This function takes the tree depth and the index of leaf node
# then build a min heap to represent the tree structure.
# The left and right ancestor nodes of a given leaf node can be
# collected separately utilizing the rationale of the minHeap

def get_ancestor_nodes(tree_depth, leaf_node):
    n_tree_nodes = (2 ** (tree_depth + 1)) - 1
    tree_heap = [_ for _ in range(n_tree_nodes)]
    # tree leaf nodes starts from 0, the corresponding index in the heap is
    # index+#branch_nodes -> index+2^d-1
    i = leaf_node + (2 ** tree_depth) - 1
    ancestor_list, ancestor_left, ancestor_right = [], [], []
    while i > 0:
        # get its parent index
        p_ind = math.floor((i - 1) / 2)
        # append parent value
        ancestor_list.append(tree_heap[p_ind])
        ## check if it is the left child of its parent
        if tree_heap[i] == tree_heap[2 * p_ind + 1]:
            ancestor_left.append(tree_heap[p_ind])
        else:
            ancestor_right.append(tree_heap[p_ind])
        # assign parent index to i, next iteration on p_ind
        i = p_ind
    return ancestor_list, ancestor_left, ancestor_right


## Distance Class ##
def get_distance_class(df_np, epsilon):
    # (1) get sorted df
    pair_index_list = np.array(list(combinations(np.arange(df_np.shape[0]), 2)))
    pair_dist_array = np.linalg.norm(
    df_np[pair_index_list[:, 0]] - df_np[pair_index_list[:, 1]], axis=1)
    pair_dist_array_index_asce = np.argsort(pair_dist_array, kind='stable')
    # # (2) get distance class based on the sorted pair df using gready method
    pair_index_list = pair_index_list.tolist()
    # DC_index_list = [[]]
    DC_Pair_list = [[]]
    curr_least_value = pair_dist_array[pair_dist_array_index_asce[0]]
    w_ind = 0
    for i in pair_dist_array_index_asce:
        if abs(curr_least_value - pair_dist_array[i]) < epsilon:
            # DC_index_list[w_ind]+=[i]
            DC_Pair_list[w_ind]+=[pair_index_list[i]]

            # if w_ind==36:
            #    print(f"{pair_dist_array[i]} - {pair_index_list[i]}")
        else:
            # DC_index_list.append([i])
            DC_Pair_list.append([pair_index_list[i]])
            w_ind+=1
            curr_least_value = pair_dist_array[i]
    # return the DC in array
    # if epsilon is 0, the first distance class would be empty set
    #  therefore, remove the empty set before returning DC
    if epsilon==0:
        return DC_Pair_list[1:]
    
    # for i in DC_Pair_list:
    #    print(i)
    # print(pair_dist_array[1])
    return DC_Pair_list

# This function takes the value of two data instance with any size
# and return the euclidian distance of the two data instance
def get_euc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_euc_distance_by_index(ind1, ind2, df):
    return get_euc_distance(df.iloc[[ind1]], df.iloc[[ind2]])


def multi_key_sorter(item):
    var0 = str(item[0])
    sub_scripts = re.findall('[0-9]+', var0)

    return (var0[0], int(sub_scripts[0]), int(sub_scripts[1]))


# input param: the model m of the optimizer, selected variable x
# this function iterate the optimal solution, and collect the results
# in a sorted list. if a specified variable is proved, only collect the
# results of the selected variable.
def get_opt_res(m, x=None):
    res_list = []
    for item in sorted([(p, m[p]) for p in m], key=multi_key_sorter):
        if x:
            if str(item[0])[0] == x:
                res_list.append(item)

        else:
            res_list.append(item)

    return res_list


# this function gets the optimal solution list and reshape the data
# in a matrix
def matrify_res(m, x, sub_script_ind=0):
    res_list = get_opt_res(m, x)
    finish_flag = False
    start_ind = 0
    node_counter = 0
    mat_res_list = []
    while not finish_flag:
        tmp_list = []

        for k in res_list[start_ind:]:
            var_sub_script = re.findall(r"[0-9]+", str(k[0]))
            if var_sub_script[sub_script_ind] == str(node_counter):
                tmp_list.append(k)
                start_ind += 1
            else:
                node_counter += 1
                mat_res_list.append(tmp_list)
                break
        if start_ind == len(res_list):
            mat_res_list.append(tmp_list)
            finish_flag = True
    return mat_res_list


# x_res is the result of boolean variable X in the matrix form (obtained from the funcion <matrify_res>)
# label_list is the list of cluster labels in ascending order
# this function maps the result of boolean variable X from boolean to corresponding labels
def get_cluster_label(x_res, label_list):
    x_res_bool_list = np.array(np.array(x_res)[..., 1], dtype=bool)
    x_res_num_list = np.sum(x_res_bool_list, axis=1) - 1
    # index of label in ascending order.
    x_pred_label = np.array([_ for _ in map(lambda x: label_list[x], x_res_num_list)])

    return x_pred_label


# This function sorts the ML and CL set in ascending order
# and return the sorted list AND corresponding index list
def sort_ascending_ML_CL(input_const, df):
    dist_list = [get_euc_distance(df.iloc[[pair_ind[0]]], df.iloc[[pair_ind[1]]]) for pair_ind in input_const]
    ascd_index = np.argsort(dist_list)
    return np.array(input_const)[ascd_index],ascd_index

def unique_encode_generator():
  num = 0
  while True:
      yield num
      num += 1

# param: the path to the constraint file
# return: ML and CL values in numpy array
def get_ML_CL(consts_path):
    # initialize empty lists for ML and CL
    list_ML,list_CL = [],[]
    with open(consts_path, 'r') as f:
        current_list = list_ML
        for line in f:
            if not line.strip():
                continue
            # ML first switch to CL when reach *
            if line.strip() == "*":
                current_list = list_CL
                continue
            # split the line into a pair of integers
            # add the pair to the current list
            current_list.append(tuple(map(int, line.split())))
    # if the ML/CL consts block is empty, the empty numpy still has to maintain the format
    array_ML = np.array(list_ML, dtype='int').reshape(-1,2) if list_ML else np.array([], dtype='int').reshape(-1,2)
    array_CL = np.array(list_CL, dtype='int').reshape(-1,2) if list_CL else np.array([], dtype='int').reshape(-1,2)
    # 
    return array_ML, array_CL


def write_clauses_to_file(f_handle, clause_list, WEIGHT):
    list_len = len(clause_list)
    if list_len>0:
        np.savetxt(f_handle, np.hstack((np.repeat(WEIGHT, list_len).reshape(-1, 1),
                                 np.array(clause_list),
                                 np.zeros(list_len).reshape(-1, 1))), fmt='%d')
def fast_write_clauses_to_file(f_handle, clause_list, WEIGHT):
    list_len = len(clause_list)
    if list_len > 0:
        fmt = ' '.join(['%g']*(clause_list.shape[1]+2))
        fmt = '\n'.join([fmt]*clause_list.shape[0]+[''])
        f_handle.write(fmt % tuple( np.hstack((np.repeat(WEIGHT, list_len).reshape(-1, 1),
                                                                    np.array(clause_list),
                                                                    np.zeros(list_len).reshape(-1, 1))).ravel()))

##
# generate and execute the loandra cmd
def loandra_solver(input_clause_file_path,
                   output_loandra_res_file_path,
                   timeout=1000, verbosity=1, PRINT_CMD=False):
  loandra_cmd = f'timeout {timeout}s ~/SAT_Project/loandra-master/loandra_static -pmreslin-cglim=30 -weight-strategy=1 -print-model -verbosity={verbosity} ' \
                + input_clause_file_path + ' >' + output_loandra_res_file_path
  if PRINT_CMD:
    print(loandra_cmd)
  os.system(loandra_cmd)

## get the result of ml and cl from the tmp ml_cl soft consts result file
def get_ml_cl_res(res_list, ml_len, cl_len):
  return res_list[-(ml_len+cl_len):-cl_len], res_list[-cl_len:]
## param: path of tmp ml_cl soft consts res file, origin ML&CL
## return the optimal feasible ml cl pairs
def get_optimal_ml_cl(res_file_name, ML, CL):
  # get res list
  with open(res_file_name) as f:
    lines_list = f.readlines()
  if lines_list[-1][0] == 'v':
    res_list = np.array(list(lines_list[-1].strip("""\nv """)), dtype='int')
  else:
    return False
  # get ml and cl res
  ml_res,cl_res = get_ml_cl_res(res_list, len(ML), len(CL))
  # ml_res = get_res(res_list, ml_var)
  # cl_res = get_res(res_list, cl_var)

  # get the final ml cl res
  ML_final_index = np.array(np.where(ml_res==1)[0])
  CL_final_index = np.array(np.where(cl_res==1)[0])

  ML_final = np.array(ML)[ML_final_index,:]
  CL_final = np.array(CL)[CL_final_index,:]

  return ML_final, CL_final

## read final res
def get_res_list(loandra_res_file_name):
  with open(loandra_res_file_name) as f:
    lines_list = f.readlines()
  if lines_list[-1][0] =='v':
    res_list = np.array(list(lines_list[-1].strip("""\nv """)), dtype='int')
    return res_list
  else:
    return False

def get_res(res_list, var0):
  if len(var0.shape) > 1:
    return res_list[var0[0,0]-1 : var0[-1,-1]]
  else:
    return res_list[var0[0]-1 : var0[-1]]
def print_MS_asec(d1, dist_list):
  dict_key_list, dict_value_list = list(d1.keys()),list(d1.values())
  for ind in np.argsort(np.array(dist_list)):
    print(f"clusters: {dict_key_list[ind]} --distance {dict_value_list[ind]}")
def print_MD_desc(d1, dist_list):
  dict_key_list, dict_value_list = list(d1.keys()),list(d1.values())
  for ind in np.argsort(-np.array(dist_list)):
    print(f"cluster: {dict_key_list[ind]} -- {dict_value_list[ind]}")


### consts by order
# This function considers ML and CL separately, the order of ML is independent of CL
def consts_kmeans_seq_sep2(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]

  kmeans_res = KMeans(n_clusters=3, random_state=1732).fit_predict(consts)
  kmeans_res_df_index = pd.DataFrame({'kmeans_res':kmeans_res}).set_index(consts_ind)
  # ml clustering results
  ml_same_cluster, ml_diff_cluster=[],[]
  for pair_ind, pair in enumerate(ML):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      ml_same_cluster.append(pair_ind)
    else:
      ml_diff_cluster.append(pair_ind)
  # cl clustering resutls
  cl_same_cluster, cl_diff_cluster=[],[]
  for pair_ind, pair in enumerate(CL):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      cl_same_cluster.append(pair_ind)
    else:
      cl_diff_cluster.append(pair_ind)
  
  # file IO
  tmp_ml_seq_list = []
  tmp_mlSame_mlDiff=np.concatenate((ml_same_cluster, ml_diff_cluster))
  for i in range(len(ML)):
    tmp_ml_seq_list.append(ml_var[tmp_mlSame_mlDiff[:i+1]])
    f.write(str(HARD_CLAUSE_W)+' '+ ' '.join(str(t) for t in ml_var[tmp_mlSame_mlDiff[:i+1]])+' 0\n')
  tmp_cl_seq_list = []
  # tmp_ml_cl_var = np.concatenate((ml_var,cl_var))
  tmp_clSame_clDiff=np.concatenate((cl_diff_cluster, cl_same_cluster))
  for i in range(len(CL)):
    tmp_cl_seq_list.append(cl_var[tmp_clSame_clDiff[:i+1]])
    f.write(str(HARD_CLAUSE_W)+' '+ ' '.join(str(t) for t in cl_var[tmp_clSame_clDiff[:i+1]])+' 0\n')

# chain link seperate <2>
def consts_kmeans_chain_sep2(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]

  kmeans_res = KMeans(n_clusters=3, random_state=1732).fit_predict(consts)
  kmeans_res_df_index = pd.DataFrame({'kmeans_res':kmeans_res}).set_index(consts_ind)
  # ml clustering results
  ml_same_cluster, ml_diff_cluster=[],[]
  for pair_ind, pair in enumerate(ML):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      ml_same_cluster.append(pair_ind)
    else:
      ml_diff_cluster.append(pair_ind)
  # cl clustering resutls
  cl_same_cluster, cl_diff_cluster=[],[]
  for pair_ind, pair in enumerate(CL):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      cl_same_cluster.append(pair_ind)
    else:
      cl_diff_cluster.append(pair_ind)
  
  # file IO
  tmp_mlSame_mlDiff=np.concatenate((ml_same_cluster, ml_diff_cluster))
  tmp_clDiff_clSame=np.concatenate((cl_diff_cluster, cl_same_cluster))
  ml_order_chain = np.vstack((tmp_mlSame_mlDiff[:-1], tmp_mlSame_mlDiff[1:])).T
  cl_order_chain = np.vstack((tmp_clDiff_clSame[:-1], tmp_clDiff_clSame[1:])).T
  for chain_pair in ml_order_chain:
    f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in ml_var[chain_pair]) + ' 0\n')
  #
  for chain_pair in cl_order_chain:
    f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in cl_var[chain_pair]) + ' 0\n')

## chain join 1
def consts_kmeans_chain_join1(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]
  kmeans_res = KMeans(n_clusters=3, random_state=1732).fit_predict(consts)
  kmeans_res_df_index = pd.DataFrame({'kmeans_res':kmeans_res}).set_index(consts_ind)
  # ml clustering results
  ml_same_cluster, ml_diff_cluster=[],[]
  for pair_ind, pair in enumerate(ML):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      ml_same_cluster.append(pair_ind)
    else:
      ml_diff_cluster.append(pair_ind)
  # cl clustering resutls
  cl_same_cluster, cl_diff_cluster=[],[]
  for pair_ind, pair in enumerate(CL):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      cl_same_cluster.append(pair_ind)
    else:
      cl_diff_cluster.append(pair_ind)
  #
  tmp_mlSame_clDiff_mlDiff_clSame = np.concatenate((ml_var[ml_same_cluster],\
                                                    cl_var[cl_diff_cluster],\
                                                    ml_var[ml_diff_cluster],\
                                                    cl_var[cl_same_cluster]))
  order_chain = np.vstack((tmp_mlSame_clDiff_mlDiff_clSame[:-1], tmp_mlSame_clDiff_mlDiff_clSame[1:])).T

  # IO
  for chain_pair in order_chain:
    f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in chain_pair) + ' 0\n')

def consts_kmeans_seq_join1(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]

  kmeans_res = KMeans(n_clusters=3, random_state=1732).fit_predict(consts)
  kmeans_res_df_index = pd.DataFrame({'kmeans_res':kmeans_res}).set_index(consts_ind)
  # ml clustering results
  ml_same_cluster, ml_diff_cluster=[],[]
  for pair_ind, pair in enumerate(ML):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      ml_same_cluster.append(pair_ind)
    else:
      ml_diff_cluster.append(pair_ind)
  # cl clustering resutls
  cl_same_cluster, cl_diff_cluster=[],[]
  for pair_ind, pair in enumerate(CL):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      cl_same_cluster.append(pair_ind)
    else:
      cl_diff_cluster.append(pair_ind)

  #
  tmp_mlSame_clDiff_mlDiff_clSame = np.concatenate((ml_var[ml_same_cluster],\
                                                    cl_var[cl_diff_cluster],\
                                                    ml_var[ml_diff_cluster],\
                                                    cl_var[cl_same_cluster]))
  order_chain = np.vstack((tmp_mlSame_clDiff_mlDiff_clSame[:-1], tmp_mlSame_clDiff_mlDiff_clSame[1:])).T
  # IO
  for i in range(len(tmp_mlSame_clDiff_mlDiff_clSame)):
    # tmp_ml_seq_list.append(ml_var[tmp_mlSame_mlDiff[:i+1]])
    f.write(str(HARD_CLAUSE_W)+' '+ ' '.join(str(t) for t in tmp_mlSame_clDiff_mlDiff_clSame[:i+1])+' 0\n')


### Distance based
# chain ml cl join 1
def consts_dist_chain_join1(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]
  tmp_ML_CL = np.vstack((ML,CL))
  tmp_ml_cl_var = np.concatenate((ml_var,cl_var))
  # inside pair distance
  tmp_ML_CL_distance = np.linalg.norm(
    np.array(df.iloc[tmp_ML_CL[:, 0]]) - np.array(df.iloc[tmp_ML_CL[:, 1]]), axis=1)
  #
  ml_cl_mixed_order = tmp_ml_cl_var[np.argsort(tmp_ML_CL_distance)]
  ml_cl_chain_join = np.vstack((ml_cl_mixed_order[:-1], ml_cl_mixed_order[1:])).T
  # IO
  for chain_pair in ml_cl_chain_join:
    f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in chain_pair) + ' 0\n')

# chain ml cl sep 2
def consts_seq_dist_chain_sep2(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]
  # ml chain
  tmp_ML_distance = np.linalg.norm(
    np.array(df.iloc[ML[:, 0]]) - np.array(df.iloc[ML[:, 1]]), axis=1)
  ml_order = ml_var[np.argsort(tmp_ML_distance)]
  ml_chain_join = np.vstack((ml_order[:-1], -ml_order[1:])).T
  # cl chain
  tmp_CL_distance = np.linalg.norm(
    np.array(df.iloc[CL[:, 0]]) - np.array(df.iloc[CL[:, 1]]), axis=1)
  cl_order = cl_var[np.argsort(tmp_CL_distance)][::-1]
  cl_chain_join = np.vstack((cl_order[:-1], -cl_order[1:])).T
  # IO
  for chain_pair in np.vstack((ml_chain_join, cl_chain_join)):
    f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in chain_pair) + ' 0\n')

# RANDOM chain ml cl sep 2
def consts_seq_random_chain_sep2(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  # ml chain
  ml_chain_join = np.vstack((ml_var[:-1], -ml_var[1:])).T
  # cl chain
  cl_chain_join = np.vstack((cl_var[:-1], -cl_var[1:])).T
  # IO
  for chain_pair in np.vstack((ml_chain_join, cl_chain_join)):
    f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in chain_pair) + ' 0\n')

# SEQUENCE ml cl sep 2
def consts_seq_dist_SEQ_sep2(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]
  # ml cl order
  tmp_ML_distance = np.linalg.norm(
    np.array(df.iloc[ML[:, 0]]) - np.array(df.iloc[ML[:, 1]]), axis=1)
  ml_order = ml_var[np.argsort(tmp_ML_distance)]
  tmp_CL_distance = np.linalg.norm(
    np.array(df.iloc[CL[:, 0]]) - np.array(df.iloc[CL[:, 1]]), axis=1)
  cl_order = cl_var[np.argsort(tmp_CL_distance)]
  # IO
  for i in range(len(ML)):
    f.write(str(HARD_CLAUSE_W)+' '+ ' '.join(str(t) for t in ml_order[:i+1])+' 0\n')
  for i in range(len(CL)):
    f.write(str(HARD_CLAUSE_W)+' '+ ' '.join(str(t) for t in cl_order[:i+1])+' 0\n')

# SEQUENCE ml cl join 1
def consts_dist_SEQ_join1(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]
  # ml cl order
  tmp_ML_distance = np.linalg.norm(
    np.array(df.iloc[ML[:, 0]]) - np.array(df.iloc[ML[:, 1]]), axis=1)
  ml_order = ml_var[np.argsort(tmp_ML_distance)]
  tmp_CL_distance = np.linalg.norm(
    np.array(df.iloc[CL[:, 0]]) - np.array(df.iloc[CL[:, 1]]), axis=1)
  cl_order = cl_var[np.argsort(tmp_CL_distance)]
  # combined
  tmp_ml_cl_order = np.concatenate((ml_order, cl_order))
  # IO
  for i in range(len(ML)+len(CL)):
    f.write(str(HARD_CLAUSE_W)+' '+ ' '.join(str(t) for t in tmp_ml_cl_order[:i+1])+' 0\n')

################## Combined Kmeans+distance
def consts_kmeans_chain_join1(df,ML, CL,ml_var, cl_var, HARD_CLAUSE_W, f, km_n_cluster):
  consts_ind = np.unique(np.vstack((ML, CL)).reshape(-1,))
  consts = df.iloc[consts_ind]
  kmeans_res = KMeans(n_clusters=km_n_cluster, random_state=1732).fit_predict(consts)
  kmeans_res_df_index = pd.DataFrame({'kmeans_res':kmeans_res}).set_index(consts_ind)
  # ml clustering results
  ml_same_cluster, ml_diff_cluster=[],[]
  for pair_ind, pair in enumerate(ML):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      ml_same_cluster.append(pair_ind)
    else:
      ml_diff_cluster.append(pair_ind)
  # cl clustering resutls
  cl_same_cluster, cl_diff_cluster=[],[]
  for pair_ind, pair in enumerate(CL):
    if kmeans_res_df_index.loc[pair[0],'kmeans_res'] == kmeans_res_df_index.loc[pair[1],'kmeans_res']:
      cl_same_cluster.append(pair_ind)
    else:
      cl_diff_cluster.append(pair_ind)
  # ml same
  tmp_ML_same_distance = np.linalg.norm(
    np.array(df.iloc[ML[ml_same_cluster, 0]]) - np.array(df.iloc[ML[ml_same_cluster, 1]]), axis=1)
  ml_same_order = ml_var[np.argsort(tmp_ML_same_distance)]
  # cl diff desc
  tmp_CL_diff_distance = np.linalg.norm(
    np.array(df.iloc[CL[cl_diff_cluster, 0]]) - np.array(df.iloc[CL[cl_diff_cluster, 1]]), axis=1)
  cl_diff_order = cl_var[np.argsort(tmp_CL_diff_distance)[::-1]]
  # ml diff
  tmp_ML_diff_distance = np.linalg.norm(
    np.array(df.iloc[ML[ml_diff_cluster, 0]]) - np.array(df.iloc[ML[ml_diff_cluster, 1]]), axis=1)
  ml_diff_order = ml_var[np.argsort(tmp_ML_diff_distance)]
  # cl diff desc
  tmp_CL_same_distance = np.linalg.norm(
    np.array(df.iloc[CL[cl_same_cluster, 0]]) - np.array(df.iloc[CL[cl_same_cluster, 1]]), axis=1)
  cl_same_order = cl_var[np.argsort(tmp_CL_same_distance)[::-1]]
  # concatenate
  tmp_mlSame_clDiff_mlDiff_clSame = np.concatenate((ml_same_order, cl_diff_order, ml_diff_order, cl_same_order))
  # format chain
  order_chain = np.vstack((tmp_mlSame_clDiff_mlDiff_clSame[:-1], tmp_mlSame_clDiff_mlDiff_clSame[1:])).T
  # IO
  for chain_pair in order_chain:
    f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in chain_pair) + ' 0\n')


#############################  euc block chain
def consts_euc_chain_block_sep(df,ML, CL,ml_var,ml_block_var, cl_var,cl_block_var, HARD_CLAUSE_W, f):
    ### sort in euc distance
    # ml chain 
    tmp_ML_distance = np.linalg.norm(
      np.array(df.iloc[ML[:, 0]]) - np.array(df.iloc[ML[:, 1]]), axis=1)
    ml_var_order = ml_var[np.argsort(tmp_ML_distance)]

    # cl chain
    tmp_CL_distance = np.linalg.norm(
      np.array(df.iloc[CL[:, 0]]) - np.array(df.iloc[CL[:, 1]]), axis=1)
    cl_var_order = cl_var[np.argsort(tmp_CL_distance)][::-1]
    
    # Blocked chain
    ml_block_chain_join = np.vstack((ml_block_var[:-1], -ml_block_var[1:])).T
    cl_block_chain_join = np.vstack((cl_block_var[:-1], -cl_block_var[1:])).T

    ### File IO for bolck enforcement
    cl_var_chain_block=np.array_split(cl_var_order,len(cl_var_order)//2)
    for chain_block_ind, chain_block in enumerate(cl_var_chain_block):
        f.write(str(HARD_CLAUSE_W)+' '+str(-cl_block_var[chain_block_ind])+' '+' '.join(str(t) for t in chain_block) + ' 0\n')

    ml_var_chain_block=np.array_split(ml_var_order,len(ml_var_order)//2)
    for chain_block_ind, chain_block in enumerate(ml_var_chain_block):
        f.write(str(HARD_CLAUSE_W)+' '+str(-ml_block_var[chain_block_ind])+' '+' '.join(str(t) for t in chain_block) + ' 0\n')
        
    ### File IO for block chain 
    
    for chain_pair in np.vstack((cl_block_chain_join, ml_block_chain_join)):
        f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in chain_pair) + ' 0\n')


#############################  euc block chain FORWARD
def write_chain_2part(consts_block_var,consts_var_chain_block, step, HARD_CLAUSE_W, f):
    # part 1
    tmp_array = []
    for block_ind, block in enumerate(consts_var_chain_block):
        tmp_array.append(np.pad(-block.reshape(-1,1), ((0,0),(1,0)), mode='constant', constant_values=consts_block_var[block_ind]).flatten())
    part1_clauses =  np.concatenate(tmp_array).reshape(-1,2)
    for clause in part1_clauses:
        f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in clause) + ' 0\n')
    # part 2
    part2_clauses = []
    for block_var, chain_block_element_array in zip(np.repeat(consts_block_var,step).reshape(-1,1), consts_var_chain_block):
         f.write(str(HARD_CLAUSE_W)+' '+' '.join(str(t) for t in np.concatenate((-block_var,chain_block_element_array))) + ' 0\n')

def consts_euc_chain_block_sep_forward(df,ML, CL,ml_var,ml_block_var, cl_var,cl_block_var, HARD_CLAUSE_W, f):
    block_size = 2
    ### sort in euc distance
    # ml chain 
    tmp_ML_distance = np.linalg.norm(
      np.array(df.iloc[ML[:, 0]]) - np.array(df.iloc[ML[:, 1]]), axis=1)
    ml_var_order = ml_var[np.argsort(tmp_ML_distance)]
    ml_var_chain_block=np.array_split(ml_var_order,len(ml_var_order)//2)

    # cl chain
    tmp_CL_distance = np.linalg.norm(
      np.array(df.iloc[CL[:, 0]]) - np.array(df.iloc[CL[:, 1]]), axis=1)
    cl_var_order = cl_var[np.argsort(tmp_CL_distance)][::-1]
    cl_var_chain_block=np.array_split(cl_var_order,len(cl_var_order)//2)

    # Blocked chain
    ml_block_chain_join = np.vstack((ml_block_var[:-1], -ml_block_var[1:])).T
    cl_block_chain_join = np.vstack((cl_block_var[:-1], -cl_block_var[1:])).T

    # write two parts of the chain to file
    write_chain_2part(ml_block_var,ml_var_chain_block,block_size, HARD_CLAUSE_W, f)
    write_chain_2part(cl_block_var,cl_var_chain_block,block_size, HARD_CLAUSE_W, f)

###### Print result output
def output_1p(loandra_res_file_name,ML,ml_var, CL, cl_var,ML_W, CL_W, TC):
  print('\nStart to read loandra res')
  loandra_status, res_list='StatusNotFund',''
  with open(loandra_res_file_name) as f:
    for line in f:
      if line[0]=='s':
            loandra_status =line[2:].strip(" \n")
      if line[0]=='v':
        res_list = np.array(list(line.split(' ')[-1][:-1]), dtype='int')

  print(f'*loandra status: {loandra_status}')
  ml_res, cl_res = np.array([]), np.array([])
  if len(res_list)>0:
    # ml and cl consts vars
    ml_res, cl_res = res_list[ml_var[0]-1 : ml_var[-1]], res_list[cl_var[0]-1 : cl_var[-1]]
  # print ml cl sat | unsat
  print(f'-ml sat and violated: {int(np.sum(ml_res))} | {len(ML) - int(np.sum(ml_res))}  | total #ml: {len(ML)} | -violated weight: {len(np.where(ml_res==0)[0])*ML_W}')
  for i in np.where(ml_res==0)[0]:
    print(f'{i} -- {ML[i]}')
  print(f'-cl sat and violated: {int(np.sum(cl_res))} | {len(CL) - int(np.sum(cl_res))}  | total #cl: {len(CL)} | -violated weight: {len(np.where(cl_res==0)[0])*CL_W}')
  for i in np.where(cl_res==0)[0]:
    print(f'{i} -- {CL[i]}')
  
  return loandra_status, ml_res, cl_res



def output_final_stage(loandra_res_file_name, b0,b1,x,y,n_points, n_labels):
  loandra_status, res_list='StatusNotFund',''
  with open(loandra_res_file_name) as f:
    for line in f:
      if line[0]=='s':
        loandra_status =line[2:].strip(" \n")
      if line[0]=='v':
        res_list = np.array(list(line.split(' ')[-1][:-1]), dtype='int')

  print(f'*loandra status: {loandra_status}')
  b0_res, b1_res = np.array([]), np.array([])
  if len(res_list)>0:
    ### objective
    b0_res, b1_res = get_res(res_list, b0), get_res(res_list, b1)
     ### ARI
    x_res_mat = res_list[x-1].reshape(n_points, -1)[:, :-1]
    label_list = np.arange(n_labels)+1
    x_pred = label_list[np.sum(x_res_mat, axis=1)-1]
    ari_odt = adjusted_rand_score(y, x_pred)
    print(f"ARI: {ari_odt}")

  print(f'b0_final: {int(np.sum(b0_res))} | 0_ind: {np.where(b0_res==0)[0][0] if len(np.where(b0_res==0)[0])>0 else 0}')
  print(f'b1_final: {int(np.sum(b1_res))} | 0_ind: {np.where(b1_res==0)[0][0] if len(np.where(b1_res==0)[0])>0 else 0}')

  return loandra_status, b0_res, b1_res

def output_final_stage_allObj(loandra_res_file_name,x,y,n_points, n_labels, b0=[],b1=[]):
  loandra_status, res_list='StatusNotFund',''
  with open(loandra_res_file_name) as f:
    for line in f:
      if line[0]=='s':
        loandra_status =line[2:].strip(" \n")
      if line[0]=='v':
        res_list = np.array(list(line.split(' ')[-1][:-1]), dtype='int')

  print(f'*loandra status: {loandra_status}')
  b0_res, b1_res = np.array([]), np.array([])
  if len(res_list)>0:
    ### objective
    if len(b0)>0:
       b0_res = get_res(res_list, b0)
    if len(b1)>0:
       b1_res = get_res(res_list, b1)
    print(f'len(b0)>0: {len(b0)>0} | len(b1)>0: {len(b1)>0}')
     ### ARI
    x_res_mat = res_list[x-1].reshape(n_points, -1)[:, :-1]
    label_list = np.arange(n_labels)+1
    x_pred = label_list[np.sum(x_res_mat, axis=1)-1]
    ari_odt = adjusted_rand_score(y, x_pred)
    print(f"ARI: {ari_odt}")

  print(f'b0_final: {int(np.sum(b0_res))} | 0_ind: {np.where(b0_res==0)[0][0] if len(np.where(b0_res==0)[0])>0 else 0}')
  print(f'b1_final: {int(np.sum(b1_res))} | 0_ind: {np.where(b1_res==0)[0][0] if len(np.where(b1_res==0)[0])>0 else 0}')

  return loandra_status, b0_res, b1_res

def output_final_stage_md(loandra_res_file_name, b0,x,y,n_points, n_labels):
  loandra_status, res_list='StatusNotFund',''
  with open(loandra_res_file_name) as f:
    for line in f:
      if line[0]=='s':
        loandra_status =line[2:].strip(" \n")
      if line[0]=='v':
        res_list = np.array(list(line.split(' ')[-1][:-1]), dtype='int')

  print(f'*loandra status: {loandra_status}')
  b0_res = np.array([])
  if len(res_list)>0:
    ### objective
    b0_res = get_res(res_list, b0)
     ### ARI
    x_res_mat = res_list[x-1].reshape(n_points, -1)[:, :-1]
    label_list = np.arange(n_labels)+1
    x_pred = label_list[np.sum(x_res_mat, axis=1)-1]
    ari_odt = adjusted_rand_score(y, x_pred)
    print(f"ARI: {ari_odt}")

  print(f'b0_final: {int(np.sum(b0_res))} | 0_ind: {np.where(b0_res==0)[0][0] if len(np.where(b0_res==0)[0])>0 else 0}')

def writeMLCL(ML, CL, outputPath):
    with open(outputPath, 'w') as f:
      for mlPair in ML:
          f.write(f"{mlPair[0]} {mlPair[1]}\n")
      f.write("*\n")
      for clPair in CL:
          f.write(f"{clPair[0]} {clPair[1]}\n")
      f.write("*\n")
      
 

## Time counter

class Time_Counter_v2:
    import time
    import numpy as np
    # np.set_printoptions(suppress=True)
    def __init__(self):
        self.time_seq_dict = {}
        self.clause_size_dict = {}

    def counter(self, input_time_start, input_time_end,action_name, df_shape=np.nan):
        self.time_seq_dict[action_name] = np.around((input_time_end - input_time_start),5)
        self.clause_size_dict[action_name] = df_shape

    def print_dict(self):
        sum_cl_time = 0
        sum_other_time = 0
        solver_time=0
        total_time=0
        for key, value in self.time_seq_dict.items():
            print(f'{key} -time: {value:.5f} -dim: {self.clause_size_dict[key]}')
            
            if bool(re.search(r'\d', key)):
                sum_cl_time += value
            elif key =="*solver*":
                solver_time=value
            elif key=="Total":
                total_time = value
        print(f'-- SUM CL TIME: {(total_time-solver_time):.5f}  \
        |  solver Time: {solver_time}  \
        |  SUM TIME: {total_time:.5f}\n')



def binary_generator():
   while True:
      yield 0
      yield 1

def write_pairs_to_file(tmp_DC, filename, type):
    if len(tmp_DC)==0:
       with open(filename, 'w') as f:
          f.write("*\n")
          f.write("*\n")
    else:
      with open(filename, 'w') as f:
          if type=="cl":
            f.write("*\n")
          for w in tmp_DC:
              for pair in w:
                  f.write(f'{pair[0]} {pair[1]}\n')
          if type=="ml":
            f.write("*\n")
          f.write("*")

def write_pairs_to_file_withBaseConsts(tmp_DC, filename, type, ML_base=[], CL_base=[]):
    
    with open(filename, 'w') as f:
      for ml_basePair in ML_base:
          f.write(f"{ml_basePair[0]} {ml_basePair[1]}\n")
      if (type=="ml" and len(tmp_DC)>0):
          for w in tmp_DC:
              for pair in w:
                  f.write(f'{pair[0]} {pair[1]}\n')
      f.write("*\n")
      for cl_basePair in CL_base:
          f.write(f"{cl_basePair[0]} {cl_basePair[1]}\n")
      if (type=="cl" and len(tmp_DC)>0):
          for w in tmp_DC:
              for pair in w:
                  f.write(f'{pair[0]} {pair[1]}\n')
      f.write("*\n")
