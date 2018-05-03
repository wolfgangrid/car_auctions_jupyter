
# coding: utf-8

# In[1]:


import numpy as np
#from scipy import interpolate
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize


def bid_fn2(f_param,f_set,f_EB,f_ES_interp):
    (f_beta,f_rho,f_w,f_J,f_alpha) = (f_param[0],f_param[1],f_param[2],f_param[3],f_param[4])
    (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (f_set[0],f_set[1],f_set[2],f_set[3],f_set[4])
    (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
    y_buy = f_y_grid.reshape(f_lyg,1) - f_b_grid.reshape(1,f_lbg)
    ES_buy = np.array([[f_ES_interp[j,s](y_buy) for s in np.arange(f_lsg)] for j in np.arange(f_J)])
    error = (f_alpha/(1-f_beta)) * f_y_grid.reshape(1,1,f_lyg,1) * (f_w.reshape(f_J,1,1,1) + f_beta*(ES_buy - f_EB.reshape(1,f_lsg,f_lyg,1))) - f_b_grid.reshape(1,1,1,f_lbg)
    return f_b_grid[np.argmin(error**2,axis=3)]


# In[11]:


def fn_B2(f_param,f_set,f_EB,f_ES_interp,bid_now,distr_b_bar_now):
    (f_beta,f_rho,f_w,f_J,f_alpha,f_sigma) = (f_param[0],f_param[1],f_param[2],f_param[3],f_param[4],f_param[5])
    (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (f_set[0],f_set[1],f_set[2],f_set[3],f_set[4])
    (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
    win_bool = (bid_now.reshape(f_J,f_lsg,f_lyg,1)>distr_b_bar_now.reshape(f_J,f_lsg,1,f_n_draws)) #dimensions: J x states x y_gridpoints x n_draws
    win_prob = np.mean(win_bool,axis=3) #dimensions: J x states x y_gridpoints
    #V1 = utility of composite good consumption plus continuation value of not winning (if inactive or active but losing the auction)
    V1 = (f_alpha*np.log((1-f_beta)*f_y_grid)).reshape(1,1,f_lyg) + (1-f_sigma*win_prob)*f_beta*f_EB.reshape(1,f_lsg,f_lyg) #dimensions: J x states x y_gridpoints
    #V2 = expected utility of winning
    y_buy = f_y_grid.reshape(1,1,f_lyg,1) - distr_b_bar_now.reshape(f_J,f_lsg,1,f_n_draws) #dimensions: J x S x Y x n_draws (b_bar)
    ES_buy = np.array([[f_ES_interp[j,s](y_buy[j,s,:,:]) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimensions: J x S x Y x n_draws (b_bar)
    V2_per_b_bar = f_w.reshape(f_J,1,1,1) - (f_alpha*distr_b_bar_now.reshape(f_J,f_lsg,1,f_n_draws) / ((1-f_beta) * f_y_grid.reshape(1,1,f_lyg,1))) + f_beta*ES_buy #dimensions: J x S x Y x n_draws (b_bar)
    V2 = f_sigma*np.mean(win_bool * V2_per_b_bar,axis=3)
    return V1 + V2

#dimensions: J x S x n_draws


# In[12]:


def fn_S2(f_param,f_set,f_EB_interp,f_ES,distr_b_2_now):
    (f_beta,f_rho,f_w,f_alpha,f_sigma) = (f_param[0],f_param[1],f_param[2],f_param[4],f_param[5])
    (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (f_set[0],f_set[1],f_set[2],f_set[3],f_set[4])
    (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
    f_J = len(f_w)
    keep = (f_w.reshape(f_J,1,1) + f_beta * f_ES) #dim: J x S x Y
    #V1 = utility of composite good consumption plus continuation value of keeping the car if not picked for auction
    V1 = (f_alpha*np.log((1-f_beta)*f_y_grid)).reshape(1,1,f_lyg) + (1-f_rho.reshape(f_J,1,1)) * keep
    #V2 = expected utility of having car picked for auction
    y_sell = f_y_grid.reshape(1,1,f_lyg,1) - distr_b_2_now.reshape(f_J,f_lsg,1,f_n_draws) #dimensions: J x S x Y x n_draws (b_2)
    EB_sell = np.array([[f_EB_interp[s](y_sell[j,s,:,:]) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimensions: J x S x Y x n_draws (b_2)
    sell_per_b_2 = (f_alpha*distr_b_2_now.reshape(f_J,f_lsg,1,f_n_draws) / ((1-f_beta) * f_y_grid.reshape(1,1,f_lyg,1))) + f_beta*EB_sell #dimensions: J x states x y_gridpoints x n_draws (b_2)
    keep_broadc = np.repeat(keep.reshape(f_J,f_lsg,f_lyg,1),f_n_draws,axis=3)
    V2_per_b_2 = np.max(np.array([keep_broadc,sell_per_b_2]),axis=0) #pick max of selling or keeping for each draw from distribution of b_2 (since seller decides to sell or not after observing the bids)
    #V2_per_b_2 = sell_per_b_2 #ignore reserve price for now
    V2 = f_rho.reshape(f_J,1,1) * np.mean(V2_per_b_2,axis=3) #integrate over distribution of b_2
    return V1 + V2


# In[13]:


def fn_iterate(f_param,f_set,
               B_start,S_start,n_iter,
              f_distr_R_states,f_transition_probs,
              f_y_realis,f_rival_bidders_ids,f_active_bidders_ids):
    (f_beta,f_rho,f_w,f_J,f_alpha,f_sigma) = (f_param[0],f_param[1],f_param[2],f_param[3],f_param[4],f_param[5])
    (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (f_set[0],f_set[1],f_set[2],f_set[3],f_set[4])
    (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
    check = []
    bids_iter = []
    Buyer_iter = []
    (B1,S1) = (B_start,S_start)
    for t in np.arange(n_iter):
        # interpolate B1 and S1
        B1_interp = np.array([[interp1d(f_y_grid,B1[j,s,:],fill_value='extrapolate') for s in np.arange(f_lsg)] for j in np.arange(f_J)])
        S1_interp = np.array([[interp1d(f_y_grid,S1[j,s,:],kind='linear',fill_value='extrapolate') for s in np.arange(f_lsg)] for j in np.arange(f_J)])
        # Calculate EB1
        distr_y_next = f_beta * f_distr_R_states.reshape(f_lsg,1,f_n_draws) * f_y_grid.reshape(1,f_lyg,1) #dimensions: state_gridpoints x y_gridpoints x n_draws
        EB_per_state_1 = np.array([[np.mean(B1_interp[j,s](distr_y_next[s,:,:]), axis=1) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimenstions: J x states x y
        EB1 = np.sum([f_rho[j]*np.dot(f_transition_probs,EB_per_state_1[j,:,:]) for j in np.arange(f_J)],axis=0) #dimensions: states x y
        # interpolate EB1
        EB1_interp = np.array([interp1d(f_y_grid,EB1[s,:], fill_value='extrapolate') for s in np.arange(f_lsg)]) #dimesions: states
        # Calculate ES1
        ES_per_state_1 = np.array([[np.mean(S1_interp[j,s](distr_y_next[s,:,:]),axis=1) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimenstions: J x states x y
        ES1 = np.array([np.dot(f_transition_probs,ES_per_state_1[j,:,:]) for j in np.arange(f_J)])
        # interpolate ES1
        ES1_interp = np.array([[interp1d(f_y_grid,ES1[j,s,:], fill_value='extrapolate') for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimenstions: J x states
        # bid function
        bid1 = bid_fn2(f_param,f_set,EB1,ES1_interp)
        # update beliefs
        distr_bids1 = np.array([[np.interp(f_y_realis[s,:],f_y_grid,bid1[j,s,:]) for s in np.arange(f_lsg)] for j in np.arange(f_J)])
        distr_b_bar_1 = np.array([[[np.sort(distr_bids1[j,s,f_rival_bidders_ids[i]])[-1] for i in np.arange(f_n_draws)] for s in np.arange(f_lsg)] for j in np.arange(f_J)])
        distr_b_2 = np.array([[[np.sort(distr_bids1[j,s,f_active_bidders_ids[i]])[-2] for i in np.arange(f_n_draws)] for s in np.arange(f_lsg)] for j in np.arange(f_J)])
        # Update Buyer Value
        B2 = fn_B2(f_param,f_set,EB1,ES1_interp,bid1,distr_b_bar_1)
        # Update Seller Value
        S2 = fn_S2(f_param,f_set,EB1_interp,ES1,distr_b_2)
        check = check + [np.absolute(B2-B1)]
        bids_iter = bids_iter + [bid1]
        Buyer_iter = Buyer_iter + [B1]
        (B1,S1) = (B2,S2)
    return [check,bids_iter,Buyer_iter]
