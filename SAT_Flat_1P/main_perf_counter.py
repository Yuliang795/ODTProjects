
import os,sys,re, time, datetime
import pandas as pd
import subprocess

# datafile list
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
        if '~' in in_data:
            tmp_df = tmp_df[~tmp_df['data'].isin(in_data)]
        else:
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
                                    in_data=["spam"],
                                    in_seed=[1732],
                                    in_kappa=[0.0]) #0.0,0.1,0.25,0.5,0.75,1.0,1.25,1.5
stage1_timeout=1800
SmartPairFlag=["smart", "nosmart"][0]
obj_ = ["mdms", "md"][0]

for consts_path in file_list:
    consts_name=consts_path.split('/')[-1]
    data_file_name = 'instance_' + consts_name.split('_')[0]
    tmp_solution_path = f'./solutions/{consts_name}_e{str(data_param_dict[data_file_name][1])}_{obj_}/'
    print(f"--------------------------------- start to execute {data_file_name} @{curr_time()}")
    ### Phase 1
    # - (1) data file path
    # - (2) tree_depth
    # - (3) epsilon
    # - (4) consts path
    # - (5) solution folder path
    # - (6) SmartPair Flag <use_SmartPair>
    # - (7) stage1 solver timeout
    # - (8) obj 
    # - output path
    ## generate cmd
    cmd = 'python3 clauses_gen.py ' +data_file_name + ' ' \
        + str(data_param_dict[data_file_name][0]) + ' ' \
        + str(data_param_dict[data_file_name][1] ) + ' ' \
        + consts_path + ' ' \
        + tmp_solution_path + ' ' \
        + SmartPairFlag   +' ' \
        + str(stage1_timeout) +' ' \
        + obj_

    #
    # create the folder for the constraints
    if not os.path.exists(tmp_solution_path):
        os.mkdir(tmp_solution_path)
    # time
    phase1_start = time.perf_counter()
    # print(cmd)
    phase1_cmd_status = subprocess.call(cmd, shell=True)
    phase1_end = time.perf_counter()
    

    print(f'{consts_name}  -finished  @{curr_time()} | '
        f'phase1 time: {phase1_end - phase1_start} | '
        )
    print(f'='*33 +"\n")

    # take a rest
    time.sleep(1)


    ## !!! CLEAN ALL CLAUSE FILES !!! ##
    cmd = f'find {tmp_solution_path} -type f -name "*clauses_final" -exec rm {{}} +'
    # print(cmd)
    os.system(cmd) 
    ## !!! CLEAN ALL DC FILES !!! ##
    cmd = f'find {tmp_solution_path} -type f -name "DC" -exec rm {{}} +'
    os.system(cmd) 

## generate csv from solutions
# cmd = f'python solution_collect.py ./solutions/ output_csv_name'
# os.system(cmd) 

