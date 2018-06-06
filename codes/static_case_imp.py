import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize
from itertools import product
import seaborn as sbn

# number of items in display
T = 6
t_w = 0.5
N = 8
size = 10
sigma = 1


# values that grid points might take
grid_space = np.linspace(10**-3, 1 - 10**-3, size)

def combs(a, r):
    """
    Return successive r-length cartesian product of values in a...
    basically just sets up the grid - I know this doesn't have to be a fucntion
    but thats how it got written in my head and I haven't changed it yet
    """
    return np.array(list(product(a, repeat=r)))

grid_values = combs(grid_space, 4)

def norm_draw(C, x):
    if C==1:
        return np.exp(-(x - 1)**2 / (2 * sigma**2))
    if C==0: 
        return np.exp(-(x)**2 / (2 * sigma**2))
    else: 
        raise ValueError('C is a binary variable and must take value 0 or 1')
    
def global_posterior(Phi, k):
    '''
    this is g_t in the write up
    '''
    phi, phi_bar, beta, beta_bar = Phi

    pres_likelihood = 1 / N * (phi * beta_bar + phi_bar * beta + (N - k) * phi_bar * beta_bar)
    abs_likelihood = phi_bar * beta_bar
    return pres_likelihood / (pres_likelihood + abs_likelihood)


def local_posterior(Phi, k):
    '''
    this is b_t,k in the write up
    '''
    phi, phi_bar, beta, beta_bar = Phi

    pres_likelihood = phi * beta_bar
    Z_b = phi * beta_bar + phi_bar * beta + \
    (N - k) * phi_bar * beta_bar  # this is the normalizing factor

    return pres_likelihood / Z_b


def p_new_ev_stay(x, Phi, sigma, k):
    '''
    this returns the probability of a new piece of evidence given
    evidence set Phi for the staying case i.e. evidence drawn from
    current location. Phi is a vector length 4.
    '''
    g_t = global_posterior(Phi, k)
    b_t = local_posterior(Phi, k)

    return (1 - g_t) * norm_draw(1, x) + g_t * (b_t * norm_draw(1, x) + (1 - b_t) * norm_draw(0, x))


def p_new_ev_switch(x, Phi, sigma, k):
    '''
    this returns the probability of a new piece of evidence given
    evidence set Phi for the switching case, Phi is a vector length 4
    '''
    phi, phi_bar, beta, beta_bar = Phi
    g_t = global_posterior(phi, phi_bar, beta, beta_bar, k)

    Z_b = phi * beta_bar + phi_bar * beta + (N - k) * phi_bar * beta_bar

    # this is the bit of the equation that captures weights the draws on local post
    # conditioned on target present
    # mostly just for notational convience
    weight_draw_pres = ((phi_bar * beta_bar) / Z_b) * norm_draw(1, x) + \
        ((phi * beta_bar + phi_bar * beta + (N - (k - 1)) * phi_bar * beta_bar) / Z_b) * norm_draw(0, x)

    return (1 - g_t) * norm_draw(0, x) + g_t * (weight_draw_pres)


def get_Update_X(Phi_t):
    '''
    takes a Phi and returns a size**4 vector of x's that update current Phi to future Phi
    it should be noted that updates only apply to the phi's and so the same update vector
    can be used for all Phi with given phi, phi_bar
    '''
    #the current phi values based on location in the grid
    phi_t = Phi_t[0]
    phi_bar_t = Phi_t[1]
    ##An initial matrix of root values to be computed for each potential 
    ##phi_tp1 and phi_bar_tp1 pair, later to be expanded for size^4 space
    init_roots = np.full((size**2,2), np.NaN)
    ##phi, phi_bar pairs 
    phi_phi_bar_space = combs(grid_space, 2)
    
    for i in range(size):
        #we look at possible values of phi_tp1 
        #!!!!!change the name
        phi_tp1 = grid_space[i]
        # the get values of X such that Phi_t is updated to each potential new Phi_t+1
        try:
            root_1 = brentq(lambda x: phi_tp1 - norm_draw(1, x) * phi_t,
                                  -150, 1)  # root finding
            root_2 = brentq(lambda x: phi_tp1 - norm_draw(1, x) * phi_t,
                                  1, 150)  # root finding
        except ValueError:
            if phi_t > phi_tp1:
                root_1 = -150
                root_2 = -150
            elif phi_t < phi_tp1:
                root_1 = 150
                root_2 = 150
        #using the root values for phi_tp1, we find the correspondent value 
        #of phi_bar_tp1 that would result from an update with the same root
        phi_bar_tp_1 = grid_space[np.abs(grid_space-(norm_draw(0, root_1)*phi_bar_t)).argmin()]
        phi_bar_tp_2 = grid_space[np.abs(grid_space-(norm_draw(0, root_2)*phi_bar_t)).argmin()]
        #based on the aquired phi_bar_tp1, we match the root_1
        # with the proper phi_tp1, phi_bar_tp1 pair 
        for j in range(size):
            if phi_phi_bar_space[(size*i)+j][1] == phi_bar_tp_1:
                init_roots[(size*i)+j][0] = root_1
            if phi_phi_bar_space[(size*i)+j][1] == phi_bar_tp_2:
                init_roots[(size*i)+j][1] = root_2

    roots = np.c_[phi_phi_bar_space, init_roots]
    # full_update = np.zeros((size**4, 2))
    # for i in range(size**2): 
    #     roots = init_roots[i, :]
    #     for j in range(size**2):
    #         full_update[(size*i)+j] = roots
    return roots

updates = get_Update_X((0.25, 0.25, 0.25, 0.25))
updates.shape

def get_Update_X_test(Phi_t):
    '''
    takes a Phi and returns a size**4 vector of x's that update current Phi to future Phi
    it should be noted that updates only apply to the phi's and so the same update vector
    can be used for all Phi with given phi, phi_bar
    '''
    #the current phi values based on location in the grid
    phi_t = Phi_t[0]
    phi_bar_t = Phi_t[1]
    ##An initial matrix of root values to be computed for each potential 
    ##phi_tp1 and phi_bar_tp1 pair, later to be expanded for size^4 space
    init_roots = np.full((size**2,2), np.NaN)
    ##phi, phi_bar pairs 
    phi_phi_bar_space = combs(grid_space, 2)
    
    for i in range(size):
        #we look at possible values of phi_tp1 
        phi_bar_tp1 = grid_space[i]
        # the get values of X such that Phi_t is updated to each potential new Phi_t+1
        try:
            root_1 = brentq(lambda x: phi_bar_tp1 - norm_draw(0, x) * phi_bar_t,
                                  -150, 0)  # root finding
            root_2 = brentq(lambda x: phi_bar_tp1 - norm_draw(0, x) * phi_bar_t,
                                  0, 150)  # root finding
        except ValueError:
            if phi_bar_t > phi_bar_tp1:
                root_1 = -150
                root_2 = -150
            elif phi_bar_t < phi_bar_tp1:
                root_1 = 150
                root_2 = 150
        #using the root values for phi_tp1, we find the correspondent value 
        #of phi_bar_tp1 that would result from an update with the same root
        phi_tp_1 = grid_space[np.abs(grid_space-(norm_draw(1, root_1)*phi_t)).argmin()]
        phi_tp_2 = grid_space[np.abs(grid_space-(norm_draw(1, root_2)*phi_t)).argmin()]
        #based on the aquired phi_bar_tp1, we match the root_1
        # with the proper phi_tp1, phi_bar_tp1 pair 
        for j in range(size):
            if phi_phi_bar_space[(size*i)+j][1] == phi_tp_1:
                init_roots[(size*i)+j][0] = root_1
            if phi_phi_bar_space[(size*i)+j][1] == phi_tp_2:
                init_roots[(size*i)+j][1] = root_2

    roots = np.c_[phi_phi_bar_space, init_roots]
    return roots

update_test = get_Update_X_test((0.8, 0.7, 0.25, 0.25))



def root_checker(phi, phi_bar, roots_arr):
    checks = np.full(size**2, False)
    for i in range(size**2):
        if np.isnan(roots_arr[i,3]):
            checks[i] = True
        else:
            phi_check = norm_draw(1, roots_arr[i,3]) * phi
            phi_bar_check = norm_draw(0, roots_arr[i,3]) * phi_bar
            checks[i] =  
            
    checks = np.c_[roots_arr[:, 0:3], checks]
    return checks

def main(argvec):
    dt, sigma, rho, reward, punishment = argvec

    R = np.array([(reward, punishment),   # (abs/abs,   abs/pres)
                  (punishment, reward)])  # (pres/abs, pres/pres) in form decision / actual


    # N x 2 matrix. First column is resp. abs, second is pres.
    decision_vals = np.zeros((size**4, 2))
    decision_vals[:, 1] = global_posterior(grid_values.flatten, np.zeros(size**4)) * R[1, 1] + (
        1 - global_posterior(grid_values.flatten, np.zeros(size**4))) * R[1, 0]  # respond present
    decision_vals[:, 0] = (1 - global_posterior(grid_values.flatten, np.zeros(size**4))) * R[0, 0] + \
        global_posterior(grid_values.flatten, np.zeros(
            size**4)) * R[0, 1]  # respond absent

    V_base = np.zeros((size**4, int(T / dt)))
    V_base[:, -1] = np.max(decision_vals, axis=1)

    decisions = np.zeros((size**4, int(T / dt)))

    # backwards induction through time
    # this emulates the code for the adhoc model
    for index in range(2, int(T / dt) + 1):
        tau = (index - 1) * dt
        t = T - tau
        print(t)

        for i in range(size**4):
            Phi = grid_values.flatten[i]
            roots = get_Update_X(Phi)
            new_phi_probs = p_new_ev_stay(roots, Phi, sigma, N)
            new_phi_probs = new_phi_probs / np.sum(new_phi_probs)
            V_stay = np.sum(new_phi_probs * V_base[:, -(index - 1)]) - rho * t

            V_base[i, -index] = np.amax((V_stay,
                                         decision_vals[i, 0], decision_vals[i, 1]))
            decisions[i, -index] = np.argmax((V_stay,
                                              decision_vals[i, 0], decision_vals[i, 1]))

    V_full = V_base.expand_dims(np.arrange(N), 3)
    decision_full = decisions.expand_dims(np.arange(N), 3)

    # backwards induction through items
'''
    for index in range(1, N):
        V_item = V_full[:, :, N - index]
        decision_item = decision_full[:, :, N - index]
        for i in range(size**4):
            for j in range(int(T / dt)):
            #V_switch =   HERE CALCULATE THE EXPECPTED VALUE FOR SWITCHING, should be analogous except with the ev_prob_switch function defined above
            if V_switch > V_item[i, j]:
                # it switch is more valuable, update the value function and the decision
                V_item[i, j] = V_switch
                decision_item[i, j] = 3
        V_full[:, :, index] = V_item
        decision_full[:, :, index] = decision_item
'''

if __name__ == '__main__':
    rho = 0.1
    reward = 1
    punishment = 0
    dt = 0.05
    sigma = 15
#    grid, times, decisions = main([dt, sigma, rho, reward, punishment])

