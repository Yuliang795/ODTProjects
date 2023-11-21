#!/bin/bash

# Print system time at the start
echo "Script started at: $(date)"

# python one_line_res_global_pomain.py ./sol.bak/T10h_res_2psmt_pareto_sep_s1732_k0.1_2.0 1732
python one_line_verify_poverify.py ./sol.bak/T10h_res_2psmt_pareto_sep_s1732_k0.1_2.0 1732
# python one_line_res_global_pomain.py ./sol.bak/T10h_res_2psmt_pareto_sep_s2352_k0.1_2.0 2352
python one_line_verify_poverify.py ./sol.bak/T10h_res_2psmt_pareto_sep_s2352_k0.1_2.0 2352
# python one_line_res_global_pomain.py ./sol.bak/T10h_res_2psmt_pareto_sep_s3556_k0.1_2.0 3556
python one_line_verify_poverify.py ./sol.bak/T10h_res_2psmt_pareto_sep_s3556_k0.1_2.0 3556


echo "Finished"