'''
Simulation Code Class Version
Written Chaitanya Baweja
Dependencies:
Mention Files
'''
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
from bayes_opt import BayesianOptimization
import time
########### Helper functions ################################

def plot_bo(f, bo):
    xs = [x["x"] for x in bo.res["all"]["params"]]
    ys = bo.res["all"]["values"]
    mean, sigma = bo.gp.predict(np.arange(len(f)).reshape(-1, 1), return_std=True)
    plt.figure(figsize=(16, 9))
    plt.plot(f)
    plt.plot(np.arange(len(f)), mean)
    plt.fill_between(np.arange(len(f)), mean+sigma, mean-sigma, alpha=0.1)
    plt.scatter(bo.X.flatten(), bo.Y, c="red", s=50, zorder=10)
    plt.xlim(0, len(f))
    plt.ylim(f.min()-0.1*(f.max()-f.min()), f.max()+0.1*(f.max()-f.min()))
    plt.show()

'''
Function to create a polynomial given coefficients:

Arguments:
----------
coefs: array of coefficients in decreasing order

Returns a polynomial with degree = lenth of coefs-1
'''
def make_poly(coefs):
    def poly(x):
        result = 0
        for c in coefs:
            result = result * x + c
        return result
    return poly
#######################################################################

class CarAuction:
    '''
    Should be made this way
    def __init__(self, args):
        self.y_grid = args.y_grid
        self.b_grid= args.b_grid
        self.beta = args.beta
        self.alpha = args.alpha
        self.w = args.w
        self.rho = args.rho
        self.sigma= args.sigma
        self.n_draws = args.n_draws
        self.N = args.N
        self.state_grid = args.state_grid
        self.J = len(self.w)
        self.lyg = args.lyg
    '''
    def __init__(self):
        y_grid = 100000 + 5000*np.arange(400)
        y_grid = y_grid.reshape(400)
        b_grid = 20000 + 5000*np.arange(0,600)
        lyg = len(y_grid)
        beta = 0.95 #discount factor
        alpha = 7.8 #parameter for value function
        w = np.array([8.3,3.4,2.2])
        param = np.array([alpha,8.3,3.4,2.2])
        J = len(w)
        rho = np.ones(np.shape(w))/np.shape(w)
        sigma=0.1 #probability of a seller being picked
        n_draws = 200
        N = 100
        state_grid = 200.0*np.arange(2,11)
        self.y_grid = y_grid
        self.b_grid= b_grid
        self.beta = beta
        self.alpha = alpha
        self.w = w
        self.rho = rho
        self.sigma= sigma
        self.n_draws = n_draws
        self.N = N
        self.state_grid = state_grid
        self.J = len(self.w)
        self.lyg = lyg

    '''
    Function for finding bid function:-

    Arguments:
    ----------

    f_param: initial parameters like beta etc
    f_set: grid, lengths etc.
    f_EB: Expected buyer value function, array of values
    f_ES_interp: Expected seller interpolation function, array of functions

    Returns grid of bid function values
    '''

    def bid_fn_y(self,f_EB_interp,f_ES_interp,y, b):
        (f_beta,f_rho,f_w,f_J,f_alpha) = (self.beta,self.rho,self.w,self.J,self.alpha)
        (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (self.y_grid,self.lyg,self.b_grid,self.n_draws,self.state_grid)
        (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
        y_buy = y - b
        ES_buy = np.array([[f_ES_interp[j,s](y_buy) for s in np.arange(f_lsg)] for j in np.arange(f_J)])
        EB_dont = np.array([[f_EB_interp[j,s](y) for s in np.arange(f_lsg)] for j in np.arange(f_J)]).reshape(f_J,f_lsg)

        error = (f_alpha/(1-f_beta)) * y * (f_w.reshape(f_J,1,1) + f_beta*(ES_buy - EB_dont)) - b
        return error**2

    def bid_fn_y2(self,f_EB_interp,f_ES_interp,y, b, j, s):
        (f_beta,f_rho,f_w,f_J,f_alpha) = (self.beta,self.rho,self.w,self.J,self.alpha)
        (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (self.y_grid,self.lyg,self.b_grid,self.n_draws,self.state_grid)
        (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
        y_buy = y - b
        ES_buy = f_ES_interp[j,s](y_buy)
        EB_dont = f_EB_interp[j,s](y)
        error = (f_alpha/(1-f_beta)) * y * f_w[j] + f_beta*(ES_buy - EB_dont) - b
        print(error)
        return -(error**2)


    def error(self, f_EB_interp,f_ES_interp):
        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}
        bo = BayesianOptimization(f= lambda b: self.bid_fn_y2(f_EB_interp,f_ES_interp,100000, b, 1, 1),
                                pbounds={"b": (0, 100000)},
                                verbose=0)

        bo.maximize(init_points=2, n_iter=25, acq="ucb", kappa=10, **gp_params)

        plot_bo(f, bo)
    '''
    Buyer Value Function:-
    Arguments:
    ----------
    f_param: initial parameters like beta etc
    f_set: grid, lengths etc.
    f_EB: Expected buyer value function, array of values
    f_ES_interp: Expected seller interpolation function, array of functions
    bid_now: value of bid functions now
    dist_b_bar_now: distribution of buyers now based on their relative bids

    Returns buyer value
    '''
    def fn_B2(self,f_EB,f_ES_interp,bid_now,distr_b_bar_now):
        (f_beta,f_rho,f_w,f_J,f_alpha,f_sigma) = (self.beta,self.rho,self.w,self.J,self.alpha,self.sigma)
        (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (self.y_grid,self.lyg,self.b_grid,self.n_draws,self.state_grid)
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
    '''
    Seller Value Function:-

    Arguments:
    ----------

    f_param: initial parameters like beta etc
    f_set: grid, lengths etc.
    f_EB_interp: Expected buyer value function(interpolation), array of functions
    f_ES: Expected seller function, array of values
    dist_b_2_now: distribution of buyers now based on their relative bids

    Returns seller value
    '''
    def fn_S2(self,f_EB_interp,f_ES,distr_b_2_now):
        (f_beta,f_rho,f_w,f_J,f_alpha,f_sigma) = (self.beta,self.rho,self.w,self.J,self.alpha,self.sigma)
        (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (self.y_grid,self.lyg,self.b_grid,self.n_draws,self.state_grid)
        (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
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
    '''
    Main function to perform iterations:-

    Arguments:
    ----------

    f_param1: Parameters like alpha, w etc.
    f_param2: Parameters like beta etc.
    f_set: all grids etc.
    B_start: initial buyer value function
    S_start: initial seller value function
    n_iter: number of iterations to perform
    f_dist_R_states:
    f_transition_probs: stock state probability transitions
    f_y_realis: personal wealth values(belief)
    f_rival_bidders_ids: rival bids
    f_active_bidders_ids: active bidders
    kind: kind of interpolation
    deg: degree of polynomial
    '''
    def fn_iterate(self, B_start,S_start,n_iter,
    f_distr_R_states,f_transition_probs, f_y_realis,f_rival_bidders_ids,
    f_active_bidders_ids,deg):
        (f_beta,f_rho,f_w, f_alpha,f_sigma) = (self.beta,self.rho,self.w,self.alpha,self.sigma)
        (f_y_grid,f_lyg,f_b_grid,f_n_draws,f_state_grid) = (self.y_grid,self.lyg,self.b_grid,self.n_draws,self.state_grid)
        f_J = len(f_w)
        (f_lyg,f_lbg,f_lsg) = (len(f_y_grid),len(f_b_grid),len(f_state_grid))
        f_param = [f_beta,f_rho,f_w,f_J,f_alpha,f_sigma]
        check = []
        bids_iter = []
        Buyer_iter = []
        (B1,S1) = (B_start,S_start)
        for t in np.arange(n_iter):
            # interpolate B1 and S1
            B1_interp = np.array([[make_poly(np.polyfit(f_y_grid,B1[j,s,:],deg=deg))for s in np.arange(f_lsg)]for j in np.arange(f_J)])
            #B1_interp = np.array([[interp1d(f_y_grid,B1[j,s,:],fill_value='extrapolate',kind=kind) for s in np.arange(f_lsg)] for j in np.arange(f_J)])
            S1_interp = np.array([[make_poly(np.polyfit(f_y_grid,S1[j,s,:],deg=deg))for s in np.arange(f_lsg)] for j in np.arange(f_J)])
            # Calculate EB1
            distr_y_next = f_beta * f_distr_R_states.reshape(f_lsg,1,f_n_draws) * f_y_grid.reshape(1,f_lyg,1) #dimensions: state_gridpoints x y_gridpoints x n_draws
            EB_per_state_1 = np.array([[np.mean(B1_interp[j,s](distr_y_next[s,:,:]), axis=1) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimenstions: J x states x y
            EB1 = np.sum([f_rho[j]*np.dot(f_transition_probs,EB_per_state_1[j,:,:]) for j in np.arange(f_J)],axis=0) #dimensions: states x y
            # interpolate EB1
            #print(np.array([np.polyfit(f_y_grid,EB1[s,:], deg=deg) for s in np.arange(f_lsg)]).shape)
            EB1_interp = np.array([make_poly(np.polyfit(f_y_grid,EB1[s,:], deg=deg)) for s in np.arange(f_lsg)]) #dimesions: states
            #print(EB1_interp)
            #EB1_interp = np.array([interp1d(f_y_grid,EB1[s,:], kind=kind, fill_value='extrapolate') for s in np.arange(f_lsg)]) #dimesions: states
            # Calculate ES1
            ES_per_state_1 = np.array([[np.mean(S1_interp[j,s](distr_y_next[s,:,:]),axis=1) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimenstions: J x states x y
            ES1 = np.array([np.dot(f_transition_probs,ES_per_state_1[j,:,:]) for j in np.arange(f_J)])
            # interpolate ES1
            ES1_interp = np.array([[make_poly(np.polyfit(f_y_grid,ES1[j,s,:], deg=deg)) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimenstions: J x states
            # print(ES1_interp)
            #ES1_interp = np.array([[interp1d(f_y_grid,ES1[j,s,:], fill_value='extrapolate',kind=kind) for s in np.arange(f_lsg)] for j in np.arange(f_J)]) #dimenstions: J x states
            #print(ES1_interp)
            # bid function
            bid1 = self.bid_fn2(EB1,ES1_interp)
            # update beliefs
            distr_bids1 = np.array([[np.interp(f_y_realis[s,:],f_y_grid,bid1[j,s,:]) for s in np.arange(f_lsg)] for j in np.arange(f_J)])
            distr_b_bar_1 = np.array([[[np.sort(distr_bids1[j,s,f_rival_bidders_ids[i]])[-1] for i in np.arange(f_n_draws)] for s in np.arange(f_lsg)] for j in np.arange(f_J)])
            distr_b_2 = np.array([[[np.sort(distr_bids1[j,s,f_active_bidders_ids[i]])[-2] for i in np.arange(f_n_draws)] for s in np.arange(f_lsg)] for j in np.arange(f_J)])
            # Update Buyer Value
            B2 = self.fn_B2(EB1,ES1_interp,bid1,distr_b_bar_1)
            # Update Seller Value
            S2 = self.fn_S2(EB1_interp,ES1,distr_b_2)
            check = check + [np.absolute(B2-B1)]
            bids_iter = bids_iter + [bid1]
            Buyer_iter = Buyer_iter + [B1]
            (B1,S1) = (B2,S2)
        return [check,bids_iter,Buyer_iter]
