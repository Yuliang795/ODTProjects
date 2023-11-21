import time,re,os,math, sys
import pandas as pd
import numpy as np
from datetime import datetime

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
ARI_r = r'ARI:\s*(-?\d+\.\d*)'

def get_var(pattern, txt, group_ind):
    try:
        return re.compile(pattern).search(txt).group(group_ind)
    except:
        return ''
def castFloat(str1):
    return float(str1) if str1 != '' else 0

solution_path = sys.argv[1]

file_name_pattern_b1 = r'out_verify_b1$'
file_name_pattern_b0 = r'out_verify_b0$'

current_date = datetime.now().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%y%m%d_%H%M")
out_folder = "./output/"+current_date+"/"
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
out_file_name=f'po_verify_{current_time}'
# out_file_handle = open(out_folder+out_file_name,'w')


verify_df_columns = ['data','seed','e','kappa','cl_ml_ratio','chain',
                     'p2_opt_lm_b0','p2_opt_lp_b1' ,'p2_lm_loandra_status','p2_lm_solver_time',
                     'p2_lp_loandra_status','p2_lp_solver_time']
out_df= pd.DataFrame(columns=verify_df_columns)

for root, folders, files in os.walk(solution_path):
        for folder in folders:
            folder_path = os.path.join(root, folder)+'/'
            
            for filename in os.listdir(folder_path):
                match_b1 = re.match(file_name_pattern_b1, filename)
                if match_b1:
                # if filename.startswith(file_name_pattern): # and filename[4:].isdigit()
                    with open(folder_path + filename, 'r+') as f:
                        line = ''.join([i for i in f])
                        p2_lp_solver_time = get_var(solver_time_r, line, 1)
                        p2_lp_clg_time = get_var(clg_time_r, line, 1)
                        p2_lp_sum_time = get_var(sum_time_r, line, 1)
                        p2_lp_b0 = get_var(b0_r, line, 1)
                        p2_lp_b1 = get_var(b1_r, line, 1)
                        p2_lp_ARI = get_var(ARI_r, line, 1)
                        p2_lp_loandra_status = get_var(loandra_status_r, line, 1)
                        


            for filename in os.listdir(folder_path):
                if re.match(file_name_pattern_b0, filename):
                    with open(folder_path + filename, 'r+') as f:
                        line = ''.join([i for i in f])
                        p2_lm_solver_time = get_var(solver_time_r, line, 1)
                        p2_lm_clg_time = get_var(clg_time_r, line, 1)
                        p2_lm_sum_time = get_var(sum_time_r, line, 1)
                        p2_lm_b0 = get_var(b0_r, line, 1)
                        p2_lm_b1 = get_var(b1_r, line, 1)
                        p2_lm_ARI = get_var(ARI_r, line, 1)
                        p2_lm_loandra_status = get_var(loandra_status_r, line, 1)

                    sol_folder_path = folder_path
                    data, kappa, seed, epsilon, cl_ml_ratio, use_chain = [re.sub(r'^(mc|s|r|e)(\d+)', r'\2', i) for i in sol_folder_path.strip(""" ./""").split('/')[-1].split('_')]

                    # b0_sum_time = '0'#b0_sum_time if b0_sum_time!= '' else '0'

                    # Create a new row
                    new_row = {}
                    
                    # Append the row to the DataFrame
                    
                    out_df.loc[len(out_df)] = [data, seed, epsilon, kappa,cl_ml_ratio,use_chain,\
                                            p2_lm_b0,p2_lp_b1,p2_lm_loandra_status, p2_lm_solver_time, \
                                            p2_lp_loandra_status, p2_lp_solver_time]

out_df.to_csv(out_folder+out_file_name, index=False)