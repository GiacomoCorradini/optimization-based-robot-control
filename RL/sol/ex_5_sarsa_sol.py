#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:56:07 2021

@author: adelprete
"""
import numpy as np
from numpy.random import randint, uniform

def sarsa(env, gamma, Q, pi, nIter, nEpisodes, maxEpisodeLength, 
          learningRate, exploration_prob, exploration_decreasing_decay,
          min_exploration_prob, compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' SARSA:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        pi: initial guess for policy
        nIter: number of iterations of the algorithm
        nEpisodes: number of episodes to be used for policy evaluation
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    h_ctg = []  # Cost-to-go during learning (for plot).
    # Make a copy of the initial Q table guess
    Q = np.copy(Q)
    nep = 0
    # for every iteration
    for k in range(nIter):
        # evaluate the current policy with TD(0)
        # for every episode
        for j in range(nEpisodes):
            nep += 1
            # reset the state
            env.reset()
            J = 0
            gamma_to_the_i = 1
            # simulate the system for maxEpisodeLength steps
            for i in range(maxEpisodeLength):  
                # with probability exploration_prob take a random control input
                if(uniform() < exploration_prob):
                    u = randint(0, env.nu)
                # otherwise take a greedy control
                else:
                    u = pi[env.x]
                x_next, cost = env.step(u)
                # Compute reference Q-value at state x
                delta = cost + gamma*Q[x_next,pi[x_next]] - Q[env.x,u]
                # Update Q-Table with the given learningRate
                Q[env.x,u] += learningRate * delta
                # keep track of the cost to go 
                J += gamma_to_the_i * cost
                gamma_to_the_i *= gamma
            h_ctg.append(J)
            # update the exploration probability with an exponential decay: eps = exp(-decay*episode)
            exploration_prob = np.exp(-exploration_decreasing_decay*nep)
            exploration_prob = max(exploration_prob, min_exploration_prob)
        # improve policy by being greedy wrt Q
        for x in range(env.nx):
            pi[x] = np.argmin(Q[env.x,:])
        # use the function compute_V_pi_from_Q(env, Q) to compute and plot V and pi
        if(k%nprint==0):
            print("Q learning - Ep %d, J=%.1f, eps=%.1f"%(k,J,100*exploration_prob))
            if(plot):
                V, pi2 = compute_V_pi_from_Q(env, Q)
                env.plot_V_table(V)
                env.plot_policy(pi2)

    return Q, h_ctg