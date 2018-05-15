'''
first go at writing the functions for static case 

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
import seaborn as sbn

#number of items in display 
T = 6
t_w = 0.5
N=8
size = 10

value_space = np.linspace(10**-3, 1-10**-3, size)

grid_values = np.zeros((size, size, size, size), dtype = tuple)
for i in range(size):
    for j in range(size):
        for y in range(size):
            for z in range(size):
                grid_values[i][j][y][z]=(value_space[i],value_space[j],value_space[y],value_space[z])
                
print(grid_values.shape)                

#phi_values = np.zeros((size, size), dtype = tuple)
#for i in range(size):
#    for j in range(size):
#        phi_values[i][j] = (value_space[i], value_space[j])

#k throughout is the number of unobserved positions

def global_posterior(Phi, k):
    '''
    this is g_t in the write up
    '''
    phi = Phi[0], phi_bar = Phi[1], beta=Phi[2], beta_bar = Phi[3]

    pres_likelihood = 1/N * (phi*beta_bar+phi_bar*beta+(N-k)*phi_bar*beta_bar)
    abs_likelihood = phi_bar*beta_bar
    return pres_likelihood/(pres_likelihood+abs_likelihood)

def local_posterior(Phi, k):
    '''
    this is b_t,k in the write up     
    '''
    phi = Phi[0], phi_bar = Phi[1], beta=Phi[2], beta_bar = Phi[3]

    pres_likelihood = phi*beta_bar
    Z_b = phi*beta_bar + phi_bar*beta + (N-k)*phi_bar*beta_bar #this is the normalizing factor
    
    return pres_likelihood/Z_b
    
def p_new_ev_stay(x, Phi, sigma, k):
    '''
    this returns the probability of a new piece of evidence given 
    evidence set Phi for the staying case i.e. evidence drawn from 
    current location. Phi is a vector length 4.
    '''
    g_t = global_posterior(Phi, k)
    b_t = local_posterior(Phi, k)
    
    draw_1 = np.exp(-(x-1)**2 / (2 * sigma**2))
    draw_0 = np.exp(-x**2 / (2 * sigma**2))
    
    return (1-g_t)*draw_0+g_t*(b_t*draw_1+(1-b_t)*draw_0)


def p_new_ev_switch(x, Phi, sigma, k):    
    '''
    this returns the probability of a new piece of evidence given 
    evidence set Phi for the staying case, Phi is a vector length 4
    '''
    phi = Phi[0], phi_bar = Phi[1], beta=Phi[2], beta_bar = Phi[3]
    g_t = global_posterior(phi, phi_bar, beta, beta_bar, k)

    Z_b = phi*beta_bar + phi_bar*beta + (N-k)*phi_bar*beta_bar
    
    draw_1 = np.exp(-(x-1)**2 / (2 * sigma**2))
    draw_0 = np.exp(-x**2 / (2 * sigma**2))
    
    ###this is the bit of the equation that captures weights the draws on local post conditioned on target present
    ###mostly just for notational convience 
    weight_draw_pres = ((phi_bar*beta_bar)/Z_b)*draw_1 + ((phi*beta_bar + phi_bar*beta + (N-(k-1))*phi_bar*beta_bar)/Z_b)*draw_0
    
    return (1-g_t)*draw_0+g_t*(weight_draw_pres)

#def get_Update_X(Phi):
#    update_Xs = np.zeros(size)
#    for i in range(size):
#        for j in range(size):
#            ## the value of X such that Phi_t is updated to each potential new Phi_t+1
#            
#            update_Xs[i] = 
#    


    
    
    
def main(argvec):
    dt, sigma, rho, reward, punishment = argvec
    
    R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)
                  (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual
    
    root_gird = np.zeros_like(grid_values, dtype = array)
    for i in range(size):
        for j in range(size):
            ## the value of X such that Phi_t is updated to each potential new Phi_t+1
            update_Xs = 
            root_grid[i, j, :, :] = updates_Xs

    decision_vals = np.zeros((size**4, 2))  # N x 2 matrix. First column is resp. abs, second is pres.
    decision_vals[:, 1] = global_posterior(grid_values.flatten, np.zeros(size**4)) * R[1, 1] + (1 -global_posterior(grid_values.flatten, np.zeros(size**4))) * R[1, 0]  # respond present
    decision_vals[:, 0] = (1 - global_posterior(grid_values.flatten, np.zeros(size**4))) * R[0, 0] + global_posterior(grid_values.flatten, np.zeros(size**4)) * R[0, 1]  # respond absent

    
        
    V_base = np.zeros((size**4, int(T/dt)))
    V_base[:, -1] = np.max(decision_vals, axis = 1)
    
    for index in range(2, int(T/dt)+1):
        tau = (index-1)*dt 
        t= T-tau 
        print(t)
        
        for i in range(size):
            Phi = grid_values.flatten[i]
            roots = root_grid[i, :]
            new_phi_probs = p_new_ev_stay(roots, Phi, sigma, N)
            new_phi_probs = new_phi_probs / np.sum(new_phi_probs)
            V_stay = np.sum(new_phi_probs * V_base[:, -(index-1)]) - rho * t 
            
            
            
            

if __name__ == '__main__':
    rho = 0.1
    reward = 1
    punishment = 0
    dt = 0.05
    sigma = 15
#    grid, times, decisions = main([dt, sigma, rho, reward, punishment])