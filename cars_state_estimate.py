import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize


from cars_model_jupyter_state import bid_fn2
from cars_model_jupyter_state import fn_B2
from cars_model_jupyter_state import fn_S2
from cars_model_jupyter_state import fn_iterate

##
# Grids
##
#y_grid = 10+30*np.array([range(0,50)]).reshape(50)
#state_grid = np.array([200,250])
#b_grid = 5*np.array(range(0,100))

y_grid = 100000 + 5000*np.arange(400)
#check if required
y_grid = y_grid.reshape(400)
b_grid = 20000 + 5000*np.arange(0,600)
lyg = len(y_grid)

##
# Parameters
##
beta = 0.95
alpha = 10
w = np.array([1,2,4])
param = np.append(alpha,w)
J = len(w)
rho = np.ones(np.shape(w))/np.shape(w)
sigma=0.1
n_draws = 200

N = 100
#b = 0.1

#import stock market level
df=pd.read_csv("/users/ridingew/car_auctions/code_jupyter/simulation_data/avgsp.csv", sep=",")
state_grid = 200.0*np.arange(2,11)
lsg = len(state_grid)

#round S&P level to nearest grid point
df['state'] = [400+200*np.round((sp-400)/200,0) for sp in df['avgsp']]
#calculate return
df['return'] = [df['avgsp'][i+1]/df['avgsp'][i] for i in range(len(df)-1)] + ['NA']


state_freq = np.array([np.sum(1*(df['state']==s))/(len(df)-1.00) for s in state_grid])
transition_probs = np.tile(state_freq,lsg).reshape(lsg,lsg)

#Realised returns in S&P Data
realis_sp_return = np.array([df['avgsp'][i+1]/df['avgsp'][i] for i in range(len(df)-1)])
#Individual Returns: one of the realised returns plus idiosyncratic noise
distr_mean_R = np.array(np.random.choice(realis_sp_return,size=n_draws,replace=True))
distr_R = distr_mean_R + np.random.uniform(-0.05,+0.05,n_draws)
distr_R_states = np.array([distr_R for s in state_grid])

y_realis_1 = np.random.lognormal(np.log(150000),np.log(2),np.round(N))

state_grid_change = np.array([state_grid[i]/state_grid[0] for i in range(lsg)])

#mean wealth
#50% of wealth invested in the stock market
mean_wealth_state = np.array([75000 + state_grid_change[s]*75000 for s in range(lsg)])

y_realis_states = np.array([np.random.lognormal(np.log(mean_wealth_state[s]),np.log(2),np.round(N)) for s in range(lsg)])

np.random.seed(2)

#initial value functions
B_cont = (1/(1-beta))*  (alpha*np.log((1-beta)*y_grid) + np.zeros(len(w)*state_grid.shape[0]*len(y_grid)).reshape(len(w),state_grid.shape[0],len(y_grid)))
S_cont = (1/(1-beta))* (alpha*np.log((1-beta)*y_grid) + np.array(np.repeat(w,state_grid.shape[0]*len(y_grid))).reshape(len(w),state_grid.shape[0],len(y_grid)))

#every one of N potential bidders has probability sigma of being active -- if m bidders are active, there are m-1 rivals
rival_bidders = np.random.binomial(N,sigma,size=n_draws) - 1
rival_bidders_ids = np.array([np.random.choice(N,size=rival_bidders[i], replace=False) for i in np.arange(n_draws)]) #dimensions: n_draws

#every one of N potential bidders has probability sigma of being active
active_bidders = np.random.binomial(N,sigma,size=n_draws)
active_bidders_ids = np.array([np.random.choice(N,size=active_bidders[i], replace=False) for i in np.arange(n_draws)])
#dimensions: n_draws

param_1 = [alpha,w]
param_2 = [beta,rho,sigma]
set1 = [y_grid,lyg,b_grid,n_draws,state_grid]

n_iter1 = 25

sim_output_test = fn_iterate(f_param1 = param_1,f_param2 = param_2,f_set = set1,
                    B_start = B_cont,S_start = S_cont, n_iter = n_iter1,
                    f_distr_R_states = distr_R_states,f_transition_probs = transition_probs,
                    f_y_realis = y_realis_states,
                    f_rival_bidders_ids=rival_bidders_ids, f_active_bidders_ids = active_bidders_ids)

sim_bid_function_test = sim_output_test[1][n_iter1-1]
sim_distr_bids_test = np.array([[np.interp(y_realis_states[s,:],y_grid,sim_bid_function_test[j,s,:]) for s in np.arange(lsg)] for j in np.arange(J)])
sim_distr_b_2_test = np.array([[[np.sort(sim_distr_bids_test[j,s,active_bidders_ids[i]])[-2] for i in np.arange(n_draws)] for s in np.arange(lsg)] for j in np.arange(J)])
sim_exp_price_test = np.mean(sim_distr_b_2_test,axis=2)
time_prices_test = np.array([interp1d(state_grid,sim_exp_price_test[j,:], fill_value='extrapolate')(df['avgsp'][1:24]) for j in range(J)])

# Data
df_prices = pd.read_csv("/users/ridingew/car_auctions/code_jupyter/simulation_data/book_total_brands.csv", sep=",")
prices_data = np.array([df_prices['avg_ferrari'],df_prices['avg_mercedes'],df_prices['avg_porsche']])


def sumsqdifference(param_est):
    alpha = param_est[0]
    w = param_est[1:]
    f_param1 = [alpha,np.array(w)]
    sim_output = fn_iterate(f_param1 = f_param1,f_param2 = param_2,f_set = set1,
                            B_start = B_cont,S_start = S_cont, n_iter = n_iter1,
                            f_distr_R_states = distr_R_states,f_transition_probs = transition_probs,
                            f_y_realis = y_realis_states,
                            f_rival_bidders_ids=rival_bidders_ids, f_active_bidders_ids = active_bidders_ids)
    sim_bid_function = sim_output[1][n_iter1-1]
    sim_distr_bids = np.array([[np.interp(y_realis_states[s,:],y_grid,sim_bid_function[j,s,:]) for s in np.arange(lsg)] for j in np.arange(J)])
    sim_distr_b_2 = np.array([[[np.sort(sim_distr_bids[j,s,active_bidders_ids[i]])[-2] for i in np.arange(n_draws)] for s in np.arange(lsg)] for j in np.arange(J)])
    sim_exp_price = np.mean(sim_distr_b_2,axis=2)
    time_prices = np.array([interp1d(state_grid,sim_exp_price[j,:], fill_value='extrapolate')(df['avgsp'][1:24]) for j in range(J)])
    diff = np.sum(np.square(prices_data - time_prices))
    return diff

alpha_start = 10
w_start = [1,2,4]
param_start = [alpha_start] + w_start

sumsqdifference(param_start)

est_output = scipy.optimize.minimize(sumsqdifference,param_start, method = 'Nelder-Mead', options={'maxiter': 1500, 'maxfev': 1500})

estimates = est_output.x

filename='/users/ridingew/car_auctions/code_jupyter/estimates1.txt'
#filename='estimates1.txt'
np.savetxt(filename, estimates)

f = open('/users/ridingew/car_auctions/code_jupyter/est_output.txt', 'w')
f.write('fun: ' + str(est_output.fun) + '\n')
f.write('message: ' + est_output.message + '\n')
f.write('nfev: ' + str(est_output.nfev) + '\n')
f.write('nit: ' + str(est_output.nit) + '\n')
f.write('status: ' + str(est_output.status) + '\n')
f.write('success: ' + str(est_output.success) + '\n')
f.write('x: ' + str(est_output.x) + '\n')
f.close()
