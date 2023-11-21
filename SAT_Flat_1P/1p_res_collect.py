import time,re,os,math, sys, datetime
import pandas as pd
import numpy as np

solution_path = sys.argv[1]


def castFloat(str1):
    return float(str1) if str1 != '' else 0

phase_1_loandra_res = 'phase_1_out_print.txt'

solver_time_r = r'\*solver\* -time: (\d+\.\d*)'
clg_time_r = r'-- SUM CL TIME: (\d+\.\d*)'
sum_time_r = r'SUM TIME: (\d+\.\d*)'
loandra_status_r = r'\*loandra status:\s*(\w+)'

# 
b0_r = r'b0_final: (\d+)'
b1_r = r'b1_final: (\d+)'
ARI_r = r'ARI:\s*(-?\d+\.\d*)'

def get_var(pattern, txt, group_ind):
    try:
        return re.compile(pattern).search(txt).group(group_ind)
    except:
        return ''




out_columns = ["data","seed","e","kappa","obj",
               "lambda_minus","lambda_plus","ARI",
               "p1_slvr_status","p1_clause_gen","p1_solver","p1_total"
               ]

out_df= pd.DataFrame(columns=out_columns)

sol_folder_counter=0
for root, folders, files in os.walk(solution_path):
    for folder in folders:
        folder_path = os.path.join(root, folder)+'/'
        with open(folder_path + phase_1_loandra_res, 'r+') as f:
            line = ''.join([i for i in f])
            p1_solver_time = get_var(solver_time_r, line, 1)
            p1_clg_time = get_var(clg_time_r, line, 1)
            p1_sum_time = get_var(sum_time_r, line, 1)
            b0 = get_var(b0_r, line, 1)
            b1 = get_var(b1_r, line, 1)
            ARI = get_var(ARI_r, line, 1)
            p1_loandra_status = get_var(loandra_status_r, line, 1)
            
            sol_folder_path = folder_path
            data, kappa,seed, epsilon,obj = [re.sub(r'^(mc|s|r|e)(\d+)', r'\2', i) for i in sol_folder_path.strip(""" ./""").split('/')[-1].split('_')]
            
            # set the clauses generation time to total system time - solver time
            p1_clg_time = str(castFloat(p1_sum_time) - castFloat(p1_solver_time))

            out_df.loc[len(out_df)]=[data, seed, epsilon, kappa,obj,\
                                    b0,b1,ARI,\
                                    p1_loandra_status, p1_clg_time, p1_solver_time, p1_sum_time
                                    ]
            
            sol_folder_counter+=1


current_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
out_file_name = f"1p_out_df_{current_time}"                
out_df.to_csv('./output/'+out_file_name,index=False)
print(f'\n***at {current_time} iterated {sol_folder_counter} res folders \n dataframe({out_df.shape}) output to ./output/{out_file_name}\n')