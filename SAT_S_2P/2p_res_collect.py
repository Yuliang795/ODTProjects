import time,re,os,math, sys,datetime
import pandas as pd
import numpy as np

solution_path = sys.argv[1]

phase_1_loandra_res = 'phase_1_out_print.txt'
phase_2_loandra_res = 'phase_2_out_print.txt'

def get_var(pattern, txt, group_ind):
    try:
        return re.compile(pattern).search(txt).group(group_ind)
    except:
        return ''
def castFloat(str1):
    return float(str1) if str1 != '' else 0

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


###
out_df_columns = ["data", "seed", "e", "kappa", "cl_ml_ratio", 
                   "ml_sat", "ml_unsat", "cl_sat", "cl_unsat",
                     "lambda_minus", "lambda_plus", "ARI", 
                     "p1_slvr_status", "p1_clause_gen", "p1_solver", "p1_total",
                    "p2_slvr_status", "p2_clause_gen", "p2_solver", "p2_total",
                    "total_time", "final_slvr_status"]
out_df= pd.DataFrame(columns=out_df_columns)

###
sol_folder_counter=0
for root, folders, files in os.walk(solution_path):
    for folder in folders:
        folder_path = os.path.join(root, folder)+'/'
            
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
        if os.path.isfile(folder_path + phase_2_loandra_res):
            with open(folder_path + phase_2_loandra_res, 'r+') as f:
                line = ''.join([i for i in f])
        else:
            line = ''
        p2_solver_time = get_var(solver_time_r, line, 1)
        p2_clg_time = get_var(clg_time_r, line, 1)
        p2_sum_time = get_var(sum_time_r, line, 1)
        b0 = get_var(b0_r, line, 1)
        b1 = get_var(b1_r, line, 1)
        ARI = get_var(ARI_r, line, 1)
        p2_loandra_status = get_var(loandra_status_r, line, 1)
    
    # print(p2_solver_time,p2_clg_time,p2_sum_time,b0,b1,ARI)

        sol_folder_path = folder_path
        data, kappa, seed,epsilon, cl_ml_ratio, useChain = [re.sub(r'^(mc|s|r|e)(\d+)', r'\2', i) for i in sol_folder_path.strip(""" ./""").split('/')[-1].split('_')]


        total_time = str(castFloat(p1_sum_time) + castFloat(p2_sum_time))
        p1_clg_time = str(castFloat(p1_sum_time) - castFloat(p1_solver_time))
        p2_clg_time = str(castFloat(p2_sum_time) - castFloat(p2_solver_time))
        if p1_loandra_status==p2_loandra_status=="OPTIMUM":
            final_slvr_status= "OPTIMUM"
        elif p1_loandra_status=="UNSATISFIABLE" or p2_loandra_status=="UNSATISFIABLE":
            final_slvr_status="UNSATISFIABLE"
        elif p1_loandra_status=="UNKNOWN" or p2_loandra_status=="UNKNOWN":
            final_slvr_status="UNKNOWN"
        elif p1_loandra_status=="SATISFIABLE" or p2_loandra_status=="SATISFIABLE":
            final_slvr_status="SATISFIABLE"
        else:
            final_slvr_status=""
        
        # ["data", "seed", "e", "kappa", "cl_ml_ratio", 
        #            "ml_sat", "ml_unsat", "cl_sat", "cl_unsat",
        #              "lambda_minus", "lambda_plus", "ARI", 
        #              "p1_slvr_status", "p1_clause_gen", "p1_solver", "p1_total",
        #             "p2_slvr_status", "p2_clause_gen", "p2_solver", "p2_total",
        #             "total_time", "final_slvr_status"]

        out_df.loc[len(out_df)]=[data, seed, epsilon, kappa,cl_ml_ratio,\
                                 ml_sat,ml_unsat,cl_sat,cl_unsat,\
                                 b0,b1,ARI,\
                                 p1_loandra_status, p1_clg_time, p1_solver_time, p1_sum_time,\
                                 p2_loandra_status, p2_clg_time, p2_solver_time, p2_sum_time,\
                                 total_time, final_slvr_status]
        
        sol_folder_counter+=1




current_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
out_file_name = f"2p_out_df_{useChain}_{current_time}"
out_df.to_csv("./output/"+out_file_name,index=False)
print(f'\n***at {current_time} iterated {sol_folder_counter} res folders \n dataframe({out_df.shape}) output to ./output/{out_file_name}\n')