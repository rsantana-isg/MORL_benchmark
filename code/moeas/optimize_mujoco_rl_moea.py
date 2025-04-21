import sys
import numpy as np
import pandas as pd

import gymnasium as gym
import mo_gymnasium as mo_gym

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2

from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO

from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import HV
from pymoo.core.problem import Problem

import matplotlib.pyplot as plt


#import psutil
import os

from tensorflow.keras import backend as K

# Parameters for the RL framework
#n_episodes = 1
#n_trials = 50






class MLP_Agent:
    def __init__(self, input_size, output_size, list_hidden_layers):        
        self.n_inputs = input_size
        self.n_outputs = output_size
        self.n_hidden = len(list_hidden_layers)
        self.list_layers = list_hidden_layers
        self.create_mlp_model()
        self.total_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])


    # Define the MLP model
    def create_mlp_model(self):
        self.model = Sequential()
    
        # Input layer
        self.model.add(Dense(self.list_layers[0], input_dim=self.n_inputs, activation='relu'))
    
        # Hidden layers
        for i in range(1,self.n_hidden):
            self.model.add(Dense(self.list_layers[i], activation='relu'))
    
        # Output layer
        self.model.add(Dense(self.n_outputs, activation='softmax'))
    
    # Function to set weights and biases from a vector
    def set_weights_from_vector(self,vector):
        vector_index = 0
        for i,layer in enumerate(self.model.layers):
            # Get the current weights and biases
            weights, biases = layer.get_weights()
        
            # Determine the shape of weights and biases
            weight_shape = weights.shape
            bias_shape = biases.shape
            
            
            # Calculate the number of elements in weights and biases
            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)
            
            # Fill weights and biases from the vector
            
            new_weights = vector[vector_index:(vector_index+weight_size)].reshape(weight_shape)
            vector_index += weight_size
            new_biases = vector[vector_index:(vector_index+bias_size)].reshape(bias_shape)
            vector_index += bias_size
        
            # Set the new weights and biases
            layer.set_weights([new_weights, new_biases])    

    def get_action(self, obs):
        """
        Returns the action predicted by the MLP agent
        """
        predictions = self.model.predict(obs,verbose=0)
        #print(predictions[0])
        action = np.random.choice(np.arange(self.n_outputs),1,p=predictions[0])[0]
        #action = np.argmax(predictions)
        return action

    
class MLP_Agent_Continuous:
    def __init__(self, input_size, output_size, list_hidden_layers):        
        self.n_inputs = input_size
        self.n_outputs = output_size
        self.n_hidden = len(list_hidden_layers)
        self.list_layers = list_hidden_layers
        self.create_mlp_model()
        self.total_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])


    # Define the MLP model
    def create_mlp_model(self):
        self.model = Sequential()
    
        # Input layer
        self.model.add(Dense(self.list_layers[0], input_dim=self.n_inputs, activation='linear'))
    
        # Hidden layers
        for i in range(1,self.n_hidden):
            self.model.add(Dense(self.list_layers[i], activation='elu'))
    
        # Output layer
        self.model.add(Dense(self.n_outputs, activation='sigmoid'))
    
    # Function to set weights and biases from a vector
    def set_weights_from_vector(self,vector):
        vector_index = 0
        for layer in self.model.layers:
            # Get the current weights and biases
            weights, biases = layer.get_weights()
        
            # Determine the shape of weights and biases
            weight_shape = weights.shape
            bias_shape = biases.shape
        
            # Calculate the number of elements in weights and biases
            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)
        
            # Fill weights and biases from the vector
            new_weights = vector[vector_index:vector_index + weight_size].reshape(weight_shape)
            vector_index += weight_size
            new_biases = vector[vector_index:vector_index + bias_size].reshape(bias_shape)
            vector_index += bias_size
        
            # Set the new weights and biases
            layer.set_weights([new_weights, new_biases])    

    def get_action(self, obs):
        """
        Returns the action predicted by the MLP agent
        """

        predictions = self.model.predict(obs,verbose=0)
        #predictions = [1 - 2*np.random.rand(self.n_outputs)]        
       
        return 1-2*predictions[0]


    
class RL_Problem_Discrete(Problem):
    def __init__(self, n_var, n_obj, agent):
        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -1 
        xu = 1  
        self.agent = agent
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=0, xl=xl, xu=xu, evaluation_of=["F"])


    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self, pop, out, *args, **kwargs):       
        all_fs = np.zeros((pop.shape[0],self.n_obj))       
        for i,x in enumerate(pop):
            K.clear_session()
            self.agent = MLP_Agent(n_inputs,n_outputs,list_layers)            
            self.agent.set_weights_from_vector(x)
           
            for j in range(n_episodes):     
                obs, info = env.reset()
                k = 0
                terminated = False
                truncated = False
                while((terminated==False) and (truncated==False) and (k<n_trials) ):                                            
                    b = tf.convert_to_tensor(obs.reshape(1,-1))
                    action = self.agent.get_action(b)    
                    obs, vector_reward, terminated, truncated, info = env.step(action)
                    if j==0:
                        tot_reward = vector_reward
                    else:
                        tot_reward = tot_reward + vector_reward                    
                    k = k + 1
                #print(i,j,vector_reward)    
            all_fs[i,:] = -tot_reward/n_episodes  #np.array([f1,f2])
            #print(i,all_fs[i,:])
            
        out["F"] = all_fs



class Single_RL_Problem_Discrete(Problem):     

    def __init__(self, n_var, n_obj, agent):
        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -1 
        xu = 1  
        self.agent = agent
        self.All_Obs = []
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=0, xl=xl, xu=xu, evaluation_of=["F"])


    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self, pop, out, *args, **kwargs):       
        all_fs = np.zeros((pop.shape[0],n_objectives))
        #process = psutil.Process(os.getpid())
        #base_memory_usage = process.memory_info().rss
        
        for i,x in enumerate(pop):
            K.clear_session()
            self.agent = agent = MLP_Agent(n_inputs,n_outputs,list_layers)                   
            self.agent.set_weights_from_vector(x)          
            for j in range(n_episodes):     
                obs, info = env.reset()
                k = 0
                terminated = False
                truncated = False
                while((terminated==False) and (truncated==False) and (k<n_trials) ):                                            
                    b = tf.convert_to_tensor(obs.reshape(1,-1))
                    action = self.agent.get_action(b)    
                    obs, vector_reward, terminated, truncated, info = env.step(action)
                    if j==0:
                        tot_reward = vector_reward
                    else:
                        tot_reward = tot_reward + vector_reward                    
                    k = k + 1
                #print(i,j,vector_reward)    
            all_fs[i,:] = -tot_reward/n_episodes  #np.array([f1,f2])
            #print(i,all_fs[i,:])

        
        coefficients = (1.0/n_objectives) * np.ones(all_fs.shape)
        
        self.All_Obs.append(all_fs)

        #print("all_fs", all_fs)
        #print("coefficients",coefficients)

        out["F"] = np.sum(np.multiply(all_fs,coefficients),1)        




        
class RL_Problem_Continuous(Problem):

    def __init__(self, n_var, n_obj, agent):
        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -1 
        xu = 1  
        self.agent = agent
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=0, xl=xl, xu=xu, evaluation_of=["F"])


    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self, pop, out, *args, **kwargs):       
        all_fs = np.zeros((pop.shape[0],self.n_obj))
        #process = psutil.Process(os.getpid())
        #base_memory_usage = process.memory_info().rss
        
        for i,x in enumerate(pop):
            K.clear_session()
            self.agent = agent = MLP_Agent_Continuous(n_inputs,n_outputs,list_layers)                   
            self.agent.set_weights_from_vector(x)
            #memory_usage = process.memory_info().rss
            #loop_memory_usage = memory_usage - base_memory_usage    
            #print(loop_memory_usage)
            for j in range(n_episodes):     
                obs, info = env.reset()
                k = 0
                terminated = False
                truncated = False
                while((terminated==False) and (truncated==False) and (k<n_trials) ):                                            
                    b = tf.convert_to_tensor(obs.reshape(1,-1))
                    action = self.agent.get_action(b)    
                    obs, vector_reward, terminated, truncated, info = env.step(action)
                    if j==0:
                        tot_reward = vector_reward
                    else:
                        tot_reward = tot_reward + vector_reward                    
                    k = k + 1
                #print(i,j,vector_reward)    
            all_fs[i,:] = -tot_reward/n_episodes  #np.array([f1,f2])
            #print(i,all_fs[i,:])
            
        out["F"] = all_fs
        

class Single_RL_Problem_Continuous(Problem):

    def __init__(self, n_var, n_obj, agent):
        # define lower and upper bounds -  1d array with length equal to number of variable
        xl = -1 
        xu = 1  
        self.agent = agent
        self.All_Obs = []
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=0, xl=xl, xu=xu, evaluation_of=["F"])


    # implemented the function evaluation function - the arrays to fill are provided directly
    def _evaluate(self, pop, out, *args, **kwargs):       
        all_fs = np.zeros((pop.shape[0],n_objectives))
        #process = psutil.Process(os.getpid())
        #base_memory_usage = process.memory_info().rss
        
        for i,x in enumerate(pop):
            K.clear_session()
            self.agent = agent = MLP_Agent_Continuous(n_inputs,n_outputs,list_layers)                   
            self.agent.set_weights_from_vector(x)
            #memory_usage = process.memory_info().rss
            #loop_memory_usage = memory_usage - base_memory_usage    
            #print(loop_memory_usage)
            for j in range(n_episodes):     
                obs, info = env.reset()
                k = 0
                terminated = False
                truncated = False
                while((terminated==False) and (truncated==False) and (k<n_trials) ):                                            
                    b = tf.convert_to_tensor(obs.reshape(1,-1))
                    action = self.agent.get_action(b)    
                    obs, vector_reward, terminated, truncated, info = env.step(action)
                    if j==0:
                        tot_reward = vector_reward
                    else:
                        tot_reward = tot_reward + vector_reward                    
                    k = k + 1
                #print(i,j,vector_reward)    
            all_fs[i,:] = -tot_reward/n_episodes  #np.array([f1,f2])
            #print(i,all_fs[i,:])

        
        coefficients = (1.0/n_objectives) * np.ones(all_fs.shape)
        
        self.All_Obs.append(all_fs)

        #print("all_fs", all_fs)
        #print("coefficients",coefficients)

        out["F"] = np.sum(np.multiply(all_fs,coefficients),1)        
        #print("F", out["F"])        


        

if __name__ == "__main__":        
    myseed = int(sys.argv[1])                # Seed: Used to set different outcomes of the stochastic program
    type_problem = int(sys.argv[2])          # Type of problem: 0: continuous action_space, 1: discrete action_space
    ind_problem =  int(sys.argv[3])          # Index of the multi-objective problem from list_problems   
    ind_algorithm = int(sys.argv[4])         # Index of the MOEA from list_MOEAS    
    n_layer_1   =  int(sys.argv[5])          # Number neurons in layer 1
    n_layer_2   =  int(sys.argv[6])          # Number neurons in layer 2
    n_layer_3   =  int(sys.argv[7])          # Number neurons in layer 3
    pop_size   =  int(sys.argv[8])           # Population size
    n_gen  =  int(sys.argv[9])               # Number of generations
    n_episodes = int(sys.argv[10])           # Number of episodes
    n_trials  = int(sys.argv[11])            # Number of trials in each episode



    n_partitions=pop_size

        
    discrete_problems = ['mo-lunar-lander-v2', 'deep-sea-treasure-v0', 'four-room-v0', 'minecart-v0']
    continuous_problems =  ['mo-hopper-v4', 'mo-halfcheetah-v4', 'mo-walker2d-v4', 'mo-ant-v4', 'mo-humanoid-v4']  # 'mo-swimmer-v4',

    list_MOEAs = [
        'MOEAD',
        'NSGA2',
        'SMSEMOA',                
        'NSGA3',
        'RNSGA2',
        'SPEA',
        '',
        '',
        '',
        '',
        'GA',
        'PSO',
        'G3PCX',
        'DE',
        'ES'
    ]

    # selection of the problem and definition of the environment
    if type_problem==0:
        mop = discrete_problems[ind_problem]
    elif type_problem==1:
        mop = continuous_problems[ind_problem]        

    env = mo_gym.make(mop)    
    obs, info = env.reset()       
    n_inputs = len(obs)    
    list_layers = [n_layer_1,n_layer_2,n_layer_3]
    
    if mop in discrete_problems:
        n_outputs =  env.action_space.n
        agent = MLP_Agent(n_inputs,n_outputs,list_layers) 
    else:
        n_outputs =  env.action_space.shape[0] 
        agent = MLP_Agent_Continuous(n_inputs,n_outputs,list_layers) 

        
    b=tf.convert_to_tensor(obs.reshape(1,-1))
    action = agent.get_action(b)    
    obs, vector_reward, terminated, truncated, info = env.step(action) # Done to compute number of objectives
    n_objectives = len(vector_reward)    
    total_params = agent.total_params
    
    algo = list_MOEAs[ind_algorithm]
        
    print("Problem:", mop)
    print("Dimensionality of the input space:", n_inputs)
    print("Number of actions:", n_outputs)
    print("Number of objectives:", n_objectives)
    print("Total number of parameters that define the neural network:", total_params)
    print("Algorithm:", algo)


    
    if mop in discrete_problems:
        if ind_algorithm>=10:   # We assume single-objective optimizers for ind_algorithm>10
            problem = Single_RL_Problem_Discrete(n_var=total_params, n_obj=1, agent=agent)
        else:    
            problem = RL_Problem_Discrete(n_var=total_params, n_obj=n_objectives, agent=agent)
    else:
        if ind_algorithm>=10:   # We assume single-objective optimizers for ind_algorithm>10
            problem = Single_RL_Problem_Continuous(n_var=total_params, n_obj=1, agent=agent)
        else:
            problem = RL_Problem_Continuous(n_var=total_params, n_obj=n_objectives, agent=agent)


    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=n_partitions)
   
    
    if algo=='MOEAD':
        ref_dirs = get_reference_directions("uniform", n_objectives, n_partitions=pop_size-1)
        algorithm = MOEAD(ref_dirs, n_neighbors=5, prob_neighbor_mating=0.7,
                          seed=myseed)
    elif algo=='NSGA2':        
        algorithm = NSGA2(pop_size=pop_size, ref_dirs=ref_dirs)
    elif algo=='SMSEMOA':        
        algorithm = SMSEMOA(pop_size=pop_size)
    elif algo=='NSGA3':  
        algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)        
    elif algo=='RNSGA2':  
        algorithm = RNSGA2(pop_size=pop_size, ref_points=ref_dirs)
    elif algo=='SPEA':          
        algorithm = SPEA2(pop_size=pop_size)
    elif algo=='GA':          
        algorithm = GA(pop_size=pop_size)
    elif algo=='PSO':          
        algorithm = PSO(pop_size=pop_size)                
    elif algo=='G3PCX':          
        algorithm = G3PCX(pop_size=pop_size)        
    elif algo=='DE':
        algorithm = DE(pop_size=pop_size, sampling=LHS(),variant="DE/rand/1/bin", CR=0.3, dither="vector", jitter=False)
    elif algo=='ES':
        algorithm = ES(n_offsprings=pop_size, rule=1.0 / 7.0)      
        

        
    out_fname = 'Enl_MOEA_results_'+str(myseed)+'_'+str(type_problem)+'_'+str(ind_problem)+'_'+str(ind_algorithm)+'_'+str(n_layer_1)+'_'+str(n_layer_2)+'_'+str(n_layer_3)+'_'+str(pop_size)+'_'+str(n_gen)+"_"+str(n_episodes)+"_"+str(n_trials)

    #out_fname = 'MOEA_results_'+str(myseed)+'_'+str(type_problem)+'_'+str(ind_problem)+'_'+str(ind_algorithm)+'_'+str(n_layer_1)+'_'+str(n_layer_2)+'_'+str(n_layer_3)+'_'+str(pop_size)+'_'+str(n_gen)
    
    print("Output file", out_fname)

    # execute the optimization
    res = minimize(problem,
                   algorithm,
                   seed=myseed,
                   termination=('n_gen', n_gen),
                   save_history=True,
                   verbose=True)

    print(res.F)

    #plot = Scatter()
    #plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    #plot.add(res.F, color="red")
    #plot.show()

    if ind_algorithm<10:
        all_hv_values = np.zeros((res.F.shape[0]))
        hv_calculator = HV(ref_point=np.max(res.F, axis=0) + 0.1)
        for i, point in enumerate(res.F):
            hv_value = hv_calculator.do(point)
            all_hv_values[i] = hv_value
            print(f'Hypervolume of {i} is {hv_value}')
    else:
        G = problem.All_Obs[-1]
        all_hv_values = np.zeros((G.shape[0]))
        hv_calculator = HV(ref_point=np.max(G, axis=0) + 0.1)
        for i in range(G.shape[0]):            
            hv_value = hv_calculator.do(G[i,:])
            all_hv_values[i] = hv_value            
            print(f'Hypervolume of {i} is {hv_value}')
            
    
    #plt.plot(G[:,0],G[:,1],'.r')    
    #plt.show()        

    
    hist = res.history

    if ind_algorithm<10:
        np.savez(out_fname, res, all_hv_values)
    else:
        np.savez(out_fname, res, all_hv_values, problem.All_Obs)
        
    # Example of how to call the program
    #python3 optimize_mujoco_rl_moea.py 111 1 1 0 4 4 4 10 2 10 50
    

   


    
