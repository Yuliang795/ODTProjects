#!/bin/bash

# Print system time at the start
echo "Script started at: $(date)"

# Command 1
python one_line_res_global_pomain.py sol.bak/t10h_mc0_correct/ 1732_mc0

# Command 2
python one_line_verify_poverify.py sol.bak/t10h_mc0_correct/ 1732_mc0

# # Command 3
# python one_line_res_global_pomain.py sol.bak/t10h_mc0_correct/ 2352_mc0

# # Command 4
# python one_line_verify_poverify.py sol.bak/t10h_mc0_correct/ 2352_mc0

# # Command 5
# python one_line_res_global_pomain.py sol.bak/t10h_mc0_correct/ 3556_mc0

# # Command 6
# python one_line_verify_poverify.py sol.bak/t10h_mc0_correct/ 3556_mc0


echo "Finished"
