
import os,sys,re, time, datetime
import pandas as pd
import subprocess

file_list = ['instance_iris', 'instance_wine', 'instance_glass',
             'instance_ionosphere', 'instance_seeds','instance_libras',
             'instance_spam', 'instance_lsun', 'instance_chainlink',
             'instance_target', 'instance_wingnut']

tree_depth_list = [3,3,4,3,3,5,3,3,3,4,3]
epsilon_list = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

data_param_dict = {key: [tree_depth, epsilon] for key, tree_depth, epsilon in zip(file_list,tree_depth_list,epsilon_list)}

def curr_time():
     return datetime.datetime.now().strftime("%H:%M:%S")

def consts_path_query(consts_df, in_data=[],in_seed=[],in_kappa=[]):
    tmp_df = consts_df
    if len(in_data)>0:
        tmp_df = tmp_df[tmp_df['data'].isin(in_data)]
    if len(in_seed)>0:
        tmp_df = tmp_df[tmp_df['seed'].isin([str(i) for i in in_seed])]
    if len(in_kappa)>0:
        tmp_df = tmp_df[tmp_df['kappa'].isin([str(i) for i in in_kappa])]
    return tmp_df.sort_values(by=['seed','data','kappa'])['path'].tolist()

consts_folder_path = './consts/'
consts_df = pd.DataFrame(columns=['data', 'kappa', 'seed', 'path'])
consts_list = []
for root, dirs, files in os.walk(consts_folder_path, topdown=False):
    for name in files:
        consts_path = os.path.join(root, name)
        consts_list.append([re.sub(r'^(s|mc|s)(\d+)',r'\2',i) for i in name.split('_')]+[consts_path])
consts_df = pd.concat([consts_df, pd.DataFrame(consts_list, columns = consts_df.columns)], ignore_index=True)

file_list = consts_path_query(consts_df,
                                    in_data=["iris"],
                                    in_seed=[1358],
                                    in_kappa=[0.0,0.1]) 
                                    # kappa cannot be 0.0 must >0
                                    #0.1,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5
use_Chain=["nochain","euc"][0]
ML_CL_ratio_set = ["1"]
stage1_timeout, stage2_timeout = [1800, 1800]
verify_b0b1=True


for ML_CL_ratio in ML_CL_ratio_set: 
    # print('*'*50 + f'Start to iterate data files with ratio{ML_CL_ratio}' + '*'*50)
    for consts_path in file_list:
        consts_name=consts_path.split('/')[-1]
        data_file_name = 'instance_' + consts_name.split('_')[0]
        tmp_solution_path = './solutions/'+f'{consts_name}_e{str(data_param_dict[data_file_name][1])}_r{ML_CL_ratio}_{use_Chain}'+'/'
        # print('='*30 + f'')
        print('='*35 + f"start to execute {consts_name} MLCL ratio: {ML_CL_ratio} @{curr_time()}" + '='*35)
        if "mc0.0" not in consts_name:
            # - (1) data file path
            # - (2) tree_depth 
            # - (3) epsilon 
            # - (4) consts path
            # - (5) solution_path 
            # - (6) CL ML ratio
            # - (7) EucChainFlag 
            # - (8) stage1 solver time out (s)
            # - (9) stage2 solver time out (s)
            # - output path
            cmd = 'python3 clauses_gen_allPhases.py ' + data_file_name + ' ' \
                + str(data_param_dict[data_file_name][0]) + ' '  \
                + str(data_param_dict[data_file_name][1] )+ ' ' \
                + consts_path  + ' ' \
                + tmp_solution_path + ' ' \
                + ML_CL_ratio +' ' \
                + use_Chain  +' ' \
                + str(stage1_timeout)  +' ' \
                + str(stage2_timeout) 
        else:
            # - (1)data file path
            # - (2) tree_depth 
            # - (3) epsilon 
            # - (4) consts 
            # - (5) solution_path 
            # - (6) final stage solver time out (s)
            # - output path
            ## For no Constarints ##
            # cmd = 'python3 clauses_gen_allPhases_noConsts.py ' + data_file_name + ' ' \
            #     + str(data_param_dict[data_file_name][0]) + ' ' \
            #     + str(data_param_dict[data_file_name][1]) + ' ' \
            #     + consts_path + ' ' \
            #     + tmp_solution_path +' ' \
            #     + str(stage2_timeout) 
            cmd = 'python3 clauses_gen_allPhases.py ' + data_file_name + ' ' \
                + str(data_param_dict[data_file_name][0]) + ' '  \
                + str(data_param_dict[data_file_name][1] )+ ' ' \
                + consts_path  + ' ' \
                + tmp_solution_path + ' ' \
                + '1' +' ' \
                + 'nochain'  +' ' \
                + str(stage1_timeout)  +' ' \
                + str(stage2_timeout) 
                
        
        #

        # create the folder for the solutions
        if not os.path.exists(tmp_solution_path):
            os.makedirs(tmp_solution_path)
        # time
        phase1_start = time.perf_counter()
        # print(cmd)
        phase1_cmd_status = subprocess.call(cmd, shell=True)
        phase1_end = time.perf_counter()
        if phase1_cmd_status!=0:
            print(f'***{curr_time()} {consts_path}\nstage-1 status error code: {phase1_cmd_status}\n')
            # sys.exit()
            continue
        print(f"finished successfully @{curr_time()}")
       

       ## Verify
        if verify_b0b1:
            cmd = 'python3 b0b1_verify.py ' + data_file_name + ' ' \
                + str(data_param_dict[data_file_name][0]) + ' ' \
                + str(data_param_dict[data_file_name][1]) + ' ' \
                + consts_path + ' ' \
                + tmp_solution_path + ' ' \
                + tmp_solution_path + 'phase_1_loandra_res'
            
            # print(cmd)
            os.system(cmd)


        time.sleep(2)



        #     ## !!! CLEAN CLAUSE FILES !!! ##
        # this one deletes all matches under current directory
        # cmd = 'find . -type f -name "*clauses_final" -exec rm {} +' 
        # this one only deletes the matches under the current solution folder
        cmd = f'find {tmp_solution_path} -type f -name "*clauses_final*" -exec rm {{}} +'
        os.system(cmd)
        ## !!! CLEAN ALL DC FILES !!! ##
        # cmd = 'find . -type f -name "DC" -exec rm {} +'
        cmd = f'find {tmp_solution_path} -type f -name "DC" -exec rm {{}} +'
        os.system(cmd)


