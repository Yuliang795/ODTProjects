import time,re,os,math, sys
from datetime import datetime
import pandas as pd
import numpy as np

phase_1_loandra_res = 'phase_1_out_print.txt'
phase_2_loandra_res = 'phase_2_out_print.txt'

solver_time_r = r'\*solver\* -time: (\d+\.\d*)'
clg_time_r = r'-- SUM CL TIME: (\d+\.\d*)'
sum_time_r = r'SUM TIME: (\d+\.\d*)'
loandra_status_r = r'\*loandra status:\s*(\w+)'
# 
ml_sat_unsat_r = r'-ml sat and violated: (\d+) \| (\d+)'
cl_sat_unsat_r = r'-cl sat and violated: (\d+) \| (\d+)'
# 
b0_r = r'b0_final: (\d+)'
b1_r = r'b1_final: (\d+)'
ARI_r = r'ARI:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'


def get_var(pattern, txt, group_ind):
    try:
        return re.compile(pattern).search(txt).group(group_ind)
    except:
        return ''
def castFloat(str1):
    return float(str1) if str1 != '' else 0

solution_path = sys.argv[1]

# file_name_pattern_b1 = r'out_(\d+)_b1$'
file_name_pattern_b0 = r'out_(\d+)_b0$'

current_date = datetime.now().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%y%m%d_%H%M")

out_folder = "./output/"+current_date+"/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
out_file_name=f'ParetoOptimal_df_'+current_time
# out_file_handle = open(out_folder+out_file_name,'w')

main_df_columns = ['data','seed','e',
                   'kappa','cl_ml_ratio','pareto_index','chain',
                   'ml_sat','ml_unsat','cl_sat','cl_unsat',
                   'lm_lambda_minus','lm_lambda_plus',
                   'lp_lambda_minus','lp_lambda_plus','ARI',
                   'p1_slvr_status','p1_clause_gen','p1_solver','p1_total',
                   'p2_lm_slvr_status','p2_lm_clause_gen','p2_lm_solver','p2_lm_total',
                   'p2_lp_slvr_status','p2_lp_clause_gen','p2_lp_solver','p2_lp_total',
                   'total_slvr_time']
out_df= pd.DataFrame(columns=main_df_columns)

for root, folders, files in os.walk(solution_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)+'/'
            if not os.path.exists(folder_path + phase_1_loandra_res):
                with open(folder_path + phase_1_loandra_res, 'w'):
                    pass
            with open(folder_path + phase_1_loandra_res, 'r+') as f:
                line = ''.join([i for i in f])
                p1_solver_time = get_var(solver_time_r, line, 1)
                p1_clg_time = get_var(clg_time_r, line, 1)
                p1_sum_time = get_var(sum_time_r, line, 1)
                ml_sat = get_var(ml_sat_unsat_r, line, 1)
                ml_unsat = get_var(ml_sat_unsat_r, line, 2)
                cl_sat = get_var(cl_sat_unsat_r, line, 1)
                cl_unsat = get_var(cl_sat_unsat_r, line, 2)
                p1_loandra_status = get_var(loandra_status_r, line, 1)
                
                # print(p1_solver_time,p1_clg_time,p1_sum_time,ml_sat,ml_unsat,cl_sat,cl_unsat)

            for filename in os.listdir(folder_path):
                match_b0 = re.match(file_name_pattern_b0, filename)
                if match_b0:
                    with open(folder_path + filename, 'r') as f:
                        line = ''.join([i for i in f])
                        p2_lm_solver_time = get_var(solver_time_r, line, 1)
                        p2_lm_clg_time = get_var(clg_time_r, line, 1)
                        p2_lm_sum_time = get_var(sum_time_r, line, 1)
                        p2_lm_b0 = get_var(b0_r, line, 1)
                        p2_lm_b1 = get_var(b1_r, line, 1)
                        p2_lm_ARI = get_var(ARI_r, line, 1)
                        if len(p2_lm_ARI)>0:
                            p2_lm_ARI = format(float(p2_lm_ARI), '.15f') 
                        p2_lm_loandra_status = get_var(loandra_status_r, line, 1)
                                
                    # if filename.startswith(file_name_pattern): # and filename[4:].isdigit()
                    file_name_b1 = f'out_{match_b0.group(1)}_b1'
                    # In the case that the file of the second stage not exists
                    try:
                        with open(folder_path + file_name_b1, 'r') as f:
                            line = ''.join([i for i in f])
                            p2_lp_solver_time = get_var(solver_time_r, line, 1)
                            p2_lp_clg_time = get_var(clg_time_r, line, 1)
                            p2_lp_sum_time = get_var(sum_time_r, line, 1)
                            p2_lp_b0 = get_var(b0_r, line, 1)
                            p2_lp_b1 = get_var(b1_r, line, 1)
                            p2_lp_ARI = get_var(ARI_r, line, 1)
                            if len(p2_lp_ARI)>0:
                                p2_lp_ARI = format(float(p2_lp_ARI), '.15f') 
                            p2_lp_loandra_status = get_var(loandra_status_r, line, 1)
                    except:
                        p2_lp_solver_time,p2_lp_clg_time,p2_lp_sum_time,\
                            p2_lp_b0,p2_lp_b1,p2_lp_ARI,p2_lp_loandra_status = ['']*7
                        
                    # if any of the b0 or b1 stage is not optimal (solver status is not Optimum), the solution
                    # will not be recorded.
                    if p2_lm_loandra_status!="OPTIMUM" or p2_lp_loandra_status!="OPTIMUM":
                        continue
                        

                    p1_clg_time = str(castFloat(p1_sum_time) - castFloat(p1_solver_time))
                    p2_lm_clg_time = str(castFloat(p2_lm_sum_time) - castFloat(p2_lm_solver_time))
                    p2_lp_clg_time = str(castFloat(p2_lp_sum_time) - castFloat(p2_lp_solver_time))

                    sol_folder_path = folder_path
                    data, kappa, seed,epsilon, cl_ml_ratio, use_chain = [re.sub(r'^(mc|s|r|e)(\d+)', r'\2', i) for i in sol_folder_path.strip(""" ./""").split('/')[-1].split('_')]

                    # b0_sum_time = '0'#b0_sum_time if b0_sum_time!= '' else '0'
                    p1_sum_time = p1_sum_time  if p1_sum_time != '' else '0'
                    one_line_res = ','.join([data, seed, epsilon, kappa,cl_ml_ratio,\
                                             match_b0.group(1), use_chain, #match_b0.group(1) is the iter digit 
                                            ml_sat,ml_unsat,cl_sat,cl_unsat,\
                                            p2_lm_b0,p2_lm_b1,p2_lp_b0,p2_lp_b1,p2_lp_ARI,\
                                            p1_loandra_status, p1_clg_time, p1_solver_time, p1_sum_time,\
                                            p2_lm_loandra_status, p2_lm_clg_time, p2_lm_solver_time, p2_lm_sum_time,\
                                            p2_lp_loandra_status, p2_lp_clg_time, p2_lp_solver_time, p2_lp_sum_time,\
            
                                            str(sum([float(t) for t in [p1_sum_time,p2_lm_solver_time,p2_lp_solver_time] if t!='']))])

                    out_df.loc[len(out_df)]=[data, seed, epsilon, kappa,cl_ml_ratio,\
                                             match_b0.group(1), use_chain, #iter digit 
                                            ml_sat,ml_unsat,cl_sat,cl_unsat,\
                                            p2_lm_b0,p2_lm_b1,p2_lp_b0,p2_lp_b1,p2_lp_ARI,\
                                            p1_loandra_status, p1_clg_time, p1_solver_time, p1_sum_time,\
                                            p2_lm_loandra_status, p2_lm_clg_time, p2_lm_solver_time, p2_lm_sum_time,\
                                            p2_lp_loandra_status, p2_lp_clg_time, p2_lp_solver_time, p2_lp_sum_time,\
                                            str(sum([float(t) for t in [p1_sum_time,p2_lm_solver_time,p2_lp_solver_time] if t!='']))]
                    # print('\n\n\n')
                    # print(one_line_res)
                    # out_file_handle.write(one_line_res+'\n')

out_df.to_csv(out_folder+out_file_name,index=False)
# out_file_handle.close()
