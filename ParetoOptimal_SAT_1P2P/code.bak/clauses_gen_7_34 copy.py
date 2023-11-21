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

def clause_gen_7_34(f,a, s, z, g, x, b0, b1, n_feature, n_points, n_labels, n_DC, tree_depth, n_bnodes, n_lnodes,
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

	### SMART PAIR clauses
	# sort ML and CL ascending
	if use_SmartPair:
		SmartPair_time_counter_start = time.perf_counter()
		#
		CL_descending,CL_descending_index = sort_ascending_ML_CL(CL, df)
		CL_descending = CL_descending[::-1].tolist()
		CL_descending_index = CL_descending_index[::-1]
		ML_ascending,ML_ascending_index = sort_ascending_ML_CL(ML, df)
		ML_ascending = ML_ascending.tolist()

		t1 = node_graph(unique_encode_generator())
		clause_list_22_23, clause_list_25_26 = [np.array([]).reshape(0, 2)] * 2
		clause_list_24 = np.array([]).reshape(0, 4)
		clause_list_27_28 = []
		clause_list_29 = []
		clause_list_b0_b1 = []
		clause_list_30_31 = []

		# np.vstack((x[pair[0], :-2], x[pair[1], :-2], x[pair[0], 1:-1],x[pair[1], 1:-1])).T


		infeasible_flag = False
		while not infeasible_flag:

				for pair in ML_ascending:
						if not t1.check_inner([pair[0], pair[1]]):
								t1.update_pos_edge([pair[0], pair[1]])
								# *add clauses (25,26) for (x_i, x_i')
								# the length of (25) and (26) are both n_label -1, so the total length is 2*n_label-2
								clause_list_25_26 = np.vstack((clause_list_25_26,
																							np.vstack((np.vstack((-x[pair[0], :-1], x[pair[1], :-1])).T,
																													np.vstack((x[pair[0], :-1], -x[pair[1], :-1])).T))
																							))

				#
				for pair in CL_descending:
						if t1.check_inner([pair[0], pair[1]]):
								print("------infeasible")
								infeasible_flag = True
								break
						if not t1.check_crossing([pair[0], pair[1]]):
								t1.update_neg_edge([pair[0], pair[1]])
								# *add clauses (22,23,24) for (x_i, x_i')
								# c22=[Or(x[pair[0]][0], x[pair[1]][0])]
								# c23=[Or(Not(x[pair[0]][-2]),
								#         Not(x[pair[1]][-2]))]
								# The length of the list of the two clauses (22), (23) is 2
								clause_list_22_23 = np.vstack((clause_list_22_23,
																							np.vstack(
																									([x[pair[0], 0], x[pair[1], 0]], [-x[pair[0], -2], -x[pair[1], -2]]))
																							))

								clause_list_24 = np.vstack((clause_list_24,
																						np.vstack((-x[pair[0], :-2], -x[pair[1], :-2], x[pair[0], 1:-1],
																											x[pair[1], 1:-1])).T
																						))

				if infeasible_flag:
						break

				E_plus_hat = copy.deepcopy(t1.E_plus)
				cc_set_hat = copy.deepcopy(t1.cc_set)
				b1w_break_flag = False
				for w in range(len(Distance_Class)):
						for pair_ind, pair in enumerate(Distance_Class[w]):
								if t1.check_crossing([pair[0], pair[1]]):
										# *add !b+w
										clause_list_b0_b1.append(-b1[w])
										# o.add(Not(b1[w]))
										b1w_break_flag = True
										print(f"-b1w_break_flag break")
										break
								if not t1.check_inner([pair[0], pair[1]]):
										t1.update_pos_edge([pair[0], pair[1]])
										# *add clause (30,31) for (x_i, x_i') and w
										# c30,c31=[],[]
										for c in range(n_labels - 1):
												# (30)
												clause_list_30_31.append([-b1[w], -x[pair[0]][c], x[pair[1]][c]])
												# clause = Or(Not(b1[w]),Not(x[pair[0]][c]), x[pair[1]][c])
												# c30 += [clause]
												# (31)
												clause_list_30_31.append([-b1[w], x[pair[0]][c], -x[pair[1]][c]])
												# clause = Or(Not(b1[w]),x[pair[0]][c], Not(x[pair[1]][c]))

										#
						if b1w_break_flag:
								print(f"--b1w_break_flag break")
								break

				#
				t1.E_plus = E_plus_hat
				t1.cc_set = cc_set_hat
				b0w_break_flag = False
				counter_p4 = 0
				for w in range(len(Distance_Class) - 1, -1, -1):
						for pair_ind, pair in enumerate(Distance_Class[w][::-1]):
								#
								if t1.check_inner([pair[0], pair[1]]):
										# *add clause b-w
										clause_list_b0_b1.append(b0[w])
										# o.add(b0[w])
										b0w_break_flag = True
										print(f"-b0 flag True")
										break
								if not t1.check_crossing([pair[0], pair[1]]):
										t1.update_neg_edge([pair[0], pair[1]])
										# *add clauses (27,28,29) for (x_i, x_i') and w
										# c27 = [Or(b0[w], x[pair[0]][0], x[pair[1]][0])]
										clause_list_27_28.append([b0[w], x[pair[0], 0], x[pair[1], 0]])
										# c28 = [Or(b0[w], Not(x[pair[0]][-2]), Not(x[pair[1]][-2]))]
										clause_list_27_28.append([b0[w], -x[pair[0], -2], -x[pair[1], -2]])
										for c in range(n_labels - 2):
												clause_list_29.append([b0[w], -x[pair[0], c], -x[pair[1], c], x[pair[0], c + 1], x[pair[1], c + 1]])

								counter_p4 += 1
						if b0w_break_flag:
								print(f"-- b0 flag break")
								break

				# finished
				infeasible_flag = True

		# write the clauses generated in smart pair to file
		clause_list_b0_b1 = np.array(clause_list_b0_b1).reshape(-1, 1)

		counter = 0
		for clause_list in [clause_list_22_23, clause_list_25_26, clause_list_24,\
												clause_list_27_28, clause_list_29,
												clause_list_30_31, clause_list_b0_b1]:
			# try:
			write_clauses_to_file(f, clause_list, HARD_CLAUSE_W)
			counter+= len(clause_list)

		# # write the clauses generated in smart pair to file

		SmartPair_time_counter_end = time.perf_counter()
		TC.counter(SmartPair_time_counter_start,
									SmartPair_time_counter_end,
									'smtPr',
									counter)


		tmp_time_counter_start = time.perf_counter()
		############## END OF SmartPair ################
	else:
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