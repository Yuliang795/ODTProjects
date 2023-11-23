import os, sys,copy, math, re, time, timeit
import numpy as np
import pandas as pd
from itertools import combinations
from functions import*
from Smart_Pair import*





# a,s,z,g,x,b0,b1 = [0]*7
# n_feature,n_points,n_labels,n_DC,tree_depth,n_bnodes,n_lnodes,feature_index = [0]*8
# HARD_CLAUSE_W,SOFT_CLAUSE_W, = [0]*3
# TC=0
# f=0
# Distance_Class=0
# df=0
# ML,CL=0,0

def clause_gen_7_21__32_34(f,a, s, z, g, x, b0, b1, n_feature, n_points, n_labels, n_DC, tree_depth, n_bnodes, n_lnodes,
                feature_index, HARD_CLAUSE_W, TC, Distance_Class, df, ML, CL, use_SmartPair):

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



	
	#####################  END OF HARD CLAUSES #######################



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