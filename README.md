# MORL_benchmark
Contains the main implementation of programs, and figures for the paper: Benchmarking MOEAs for solving continuous multi-objective RL problems (https://arxiv.org/abs/).

This project implements single-objective evolutionary algorithms and  multi-objective evolutionary algorithms (MOEAs) for solving complex multi-objective reinforcement learning (MORL) problems. The code allows to solve the following MORL instances: mo-ant-v4, mo-humanoid-v4, mo-halfcheetah-v4, mo-walker2d-v4, and mo-hopper-v4. 

After running the code it is possible to evaluate different metrics of MOEA behavior: GD, GDPlus,  HV,  IGD, and IGDPlus. The values of each metric through generations can be visualized, as well as the approximated Pareto fronts (PFs) and scalarized objectives (same weights for all objectives). 

code folder contains the code used to run MOEAs and for postprocessing the results
metrics folders contains the values of the metrics computed at each generation for each experiments
figures folder contains the figures of the metrics computed for all MOEAs and MORL instances 


To run any of the algorithms with the MORL instances, check input variables in file code/moeas/optimize_mujoco_rl_moea.py

One example of how to run the code is included with comments at the end of file code/moeas/optimize_mujoco_rl_moea.py
 
 

Citation 

InProceedings{Hernandez_and_Santana:2025,
  title={Benchmarking {MOEAs} for solving continuous multi-objective {RL} problems },
  author={Hernandez, Carlos and Santana, Roberto},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference (GECCO-2025)},
  pages={},
  year={2022},
  note={Accepted for publication}
}


