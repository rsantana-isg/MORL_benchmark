import sys
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt



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
discrete_problems = ['mo-lunar-lander-v2', 'deep-sea-treasure-v0', 'four-room-v0', 'minecart-v0']
continuous_problems =  ['mo-hopper-v4', 'mo-halfcheetah-v4', 'mo-walker2d-v4', 'mo-ant-v4', 'mo-humanoid-v4']  # 'mo-swimmer-v4',



def Read_Metrics(fname, gen_val):
    csv_fields = ['algorithm','experiment','generation','Hypervolume','GD','IGD','GDPlus','IGDPlus','Scalarized-objective']
    # Load the CSV file

    df = pd.read_csv(fname)

    
    headers = ['PSO','NSGA2','SMSEMOA', 'RNSGA2' , 'SPEA']
    
    # Filter the DataFrame where 'generation' equals gen_val
    filtered_df = df[df['generation'] == gen_val]

    print(filtered_df)
    # Group by 'algorithm' and compute the mean for 'Hypervolume' and 'IGD'
    mean_values = filtered_df.groupby('algorithm')[['Hypervolume', 'IGD']].mean()
    std_values = filtered_df.groupby('algorithm')[['Hypervolume', 'IGD']].std()
    print(mean_values)
    print(std_values)
    from tabulate import tabulate

    #indices = [4,2,6,5,7]
    indices = [1,0,3,2,4]
    A = mean_values.values.transpose()[0,indices]
    B = mean_values.values.transpose()[1,indices]
    D = std_values.values.transpose()[0,indices]
    E = std_values.values.transpose()[1,indices]

    for i in range(5):
        if i==0:
            C = np.hstack((A[i],D[i]))
            F = np.hstack((B[i],E[i]))
        else:
            C = np.hstack((C,np.hstack((A[i],D[i]))))
            F = np.hstack((F,np.hstack((B[i],E[i]))))

    all_parts = np.hstack((C,F))   
    table = ['',all_parts]
    
    print(tabulate(table, tablefmt="latex", floatfmt=".3f"))
  
    # Display the resulting means

   
    # Optionally, save the results to a CSV file
    #mean_values.to_csv('mean_results.csv', index=True)

if __name__ == "__main__":
    mujoco_problem = int(sys.argv[1])
    n_episodes = int(sys.argv[2])
    n_layer_2 =  int(sys.argv[3])
    n_gen =  int(sys.argv[4])   # 24,99
    gval = int(sys.argv[5]) # 24 99
    
    the_algs = [0,1,2,3,4,5,10,11,13]
    #the_algs = [4,5,11,1,2]
    #the_algs = [10]
    
    the_seeds= [0,1,2,3,4,5,6,7,8,9,10]
    n_algs = len(the_algs)    
    n_experiments = len(the_seeds)
        
    all_hv = np.zeros((n_algs,n_gen))
    All_F = []    
    fname =  'Metrics_With_MOEAD/Metrics_'+ continuous_problems[mujoco_problem]+'_'+str(n_episodes)+'_'+str(n_layer_2)+'_'+str(n_gen)+'.csv'
    print(fname)
    Read_Metrics(fname,gval)


    
   
                                    
                
   
        
