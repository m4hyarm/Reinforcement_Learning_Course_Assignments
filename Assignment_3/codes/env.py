import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gym import Env
import time


def make_map(shape, studentNum, safe_break_prob, random_break): 
    
    shape = np.array(shape)
    minmoves = sum(shape)-2
    np.random.seed(studentNum)  
    move = np.zeros(minmoves)  # Minimum moves for start to the end point
    idx = np.random.choice(range(minmoves),size=minmoves//2,replace=False)
    move[idx] = 1

    point = [0,0]
    lowprobs = [tuple(point)]

    for m in move:
        if m:
            point[0] += 1
        else:
            point[1] += 1
        lowprobs.append(tuple(point))

    idx = np.array(lowprobs)
    map = np.ones(shape)
    if random_break:
        map *= np.random.rand(shape[0],shape[1])
    map[idx[:,0],idx[:,1]] = safe_break_prob
    map[0,0] = 0.0                     # Start point
    map[shape[0]-1,shape[1]-1] = 0.0   # End point
    
    safe_path = np.zeros(shape)
    safe_path[idx[:,0],idx[:,1]] = 1
    safe_path[0,0] = 1
    safe_path[shape[0]-1,shape[1]-1] = 1
    
    return map, safe_path


class FrozenLake(Env):
    def __init__(self, actionSpace, Map, notSlipProb, goalReward, moveReward, failReward, discount_factor, theta):
        self.Discount    = discount_factor
        self.theta       = theta
        self.actionSpace = actionSpace
        self.map         = Map
        self.reshapedMap = Map.reshape(-1)
        self.numstates   = Map.size
        self.numactions  = len(actionSpace)
        self.notSlipProb = notSlipProb
        self.V           = np.zeros(self.numstates)
        self.probability = np.full((self.numstates, self.numactions), fill_value=0.25)
        for s in range(self.numstates):
            if self.reshapedMap[s] == 1 or s == self.numstates-1:
                self.probability[s] = 0
        
        # transition space
        colNum           = Map.shape[1]
        self.transitions = np.zeros((self.numstates, self.numactions, self.numstates))
        for a in self.actionSpace:
            for s in range(self.numstates):
                if ((s//colNum == 0) & (a == 3)) or (((self.numstates-s-1)//colNum == 0) & (a == 1)) or \
                   ((s%colNum  == 0) & (a == 0)) or ((s%colNum  == (colNum-1)) & (a == 2)):
                    self.transitions[s, a, s] = 1

                elif a == 0:
                    self.transitions[s, a, s-1] = 1

                elif a == 1:
                    self.transitions[s, a, s+colNum] = 1

                elif a == 2:
                    self.transitions[s, a, s+1] = 1

                elif a == 3:
                    self.transitions[s, a, s-colNum] = 1
                    
                if self.reshapedMap[s] == 1 or s == self.numstates-1:
                    self.transitions[s, a, :] = 0
                    self.transitions[s, a, s] = 1

        # next states
        self.nextStates  = np.zeros((self.numstates, self.numactions), dtype=int)
        for s in range(self.numstates):
            self.nextStates[s] = np.argmax(self.transitions[s], axis=1)

            
        # defining fail probabilities
        for s, nexts in enumerate(self.nextStates):
            if np.all(nexts == nexts[0]) != True:
                for a, next_s in enumerate(nexts):
                    self.transitions[s, a, next_s] = self.notSlipProb
                    slips = set(nexts) - set([next_s])
                    self.transitions[s, a, list(slips)] = (1-self.notSlipProb)/len(slips)
                    
        
        # rewards & fail probabilities
        self.rewards    = np.zeros((self.numstates, self.numstates))
        self.pfail      = np.zeros((self.numstates, self.numstates))
        self.failReward = failReward
        for s, nexts in enumerate(self.nextStates):
            for next_s in nexts:
                self.rewards[s, next_s] = moveReward
                self.pfail[s, next_s] = 1
                if next_s == self.numstates-1:
                    self.rewards[s, self.numstates-1] += goalReward
                if s == next_s:
                    self.rewards[s, next_s] = 0
                    self.pfail[s, next_s] = 0
            self.pfail[s] *= self.reshapedMap


    def reset(self):
        self.V           = np.zeros(self.numstates)
        self.probability = np.full((self.numstates, self.numactions), fill_value=0.25)
        for s in range(self.numstates):
            if self.reshapedMap[s] == 1 or s == self.numstates-1:
                self.probability[s] = 0
        

    def step(self, state, action):
        nextState = self.nextStates[state, action]
        reward = self.rewards[state, nextState]
        if nextState == self.numstates-1: Goal=True
        else: Goal=False
        print(f'Next State: {nextState} \nReward of Action: {reward} \nReach Goal: {Goal}')
        
        
    def Policy_Evaluation(self):
        self.counter = 0
        while True:
            self.counter += 1
            delta = 0
            for state in range(self.numstates):

                v = self.V[state].copy()
                q = np.sum(self.transitions[state] *\
                           ((1-self.pfail[state]) * (self.rewards[state] + self.Discount * np.tile(self.V, (self.numactions, 1))) +\
                            (np.tile(self.pfail[state], (self.numactions, 1)) * (self.rewards[state]+self.failReward)) ), axis=1)
                self.V[state] = np.sum(self.probability[state] * q)
                delta = max(delta, np.abs(v - self.V[state]))

            if delta < self.theta:
                break
            
        return self.counter    
        
    def Policy_Improvment(self):
        policy_stable = True
        for state in range(self.numstates):
            if self.reshapedMap[state] == 1 or state == self.numstates-1:
                continue

            old = self.probability[state].copy()
            q = np.sum(self.transitions[state] *\
                       ((1-self.pfail[state]) * (self.rewards[state] + self.Discount * np.tile(self.V, (self.numactions, 1))) +\
                        (np.tile(self.pfail[state], (self.numactions, 1)) * (self.rewards[state]+self.failReward)) ), axis=1)
            opt_act = np.argmax(q)
            self.probability[state] = np.eye(self.numactions)[opt_act]
            if (self.probability[state] != old).any():
                policy_stable = False
                
        return policy_stable
            
        
    def Policy_Iteration(self):
        self.reset()
        counter = 0
        start = time.time()
        while True:
            counter += 1
            eval_iterations = self.Policy_Evaluation()
            policy_stable = self.Policy_Improvment()
            print(f'Iteration: {counter} \nNumber of Evaluation Iterations: {eval_iterations}\n------------------------------------')
            
            if policy_stable:
                end = time.time()
                print(f'\nTime Elapsed: {(end - start):.3f}s')
                break
            
            
    def Value_Iteration(self):
        self.reset()
        start = time.time()
        counter = 0
        while True:
            counter += 1
            delta = 0
            for state in range(self.numstates):

                v = self.V[state].copy()
                q = np.sum(self.transitions[state] *\
                           ((1-self.pfail[state]) * (self.rewards[state] + self.Discount * np.tile(self.V, (self.numactions, 1))) +\
                            (np.tile(self.pfail[state], (self.numactions, 1)) * (self.rewards[state]+self.failReward)) ), axis=1)
                self.V[state] = np.max(q)
                delta = max(delta, np.abs(v - self.V[state]))

            if delta < self.theta:
                break

        # policy improvement
        for state in range(self.numstates):
            if self.reshapedMap[state] == 1 or state == self.numstates-1:
                continue

            q = np.sum(self.transitions[state] *\
                       ((1-self.pfail[state]) * (self.rewards[state] + self.Discount * np.tile(self.V, (self.numactions, 1))) +\
                        (np.tile(self.pfail[state], (self.numactions, 1)) * (self.rewards[state]+self.failReward)) ), axis=1)
            opt_act = np.argmax(q)
            self.probability[state] = np.eye(self.numactions)[opt_act]
            
        end = time.time()
        print(f'Number of Iterations: {counter} \nTime Elapsed: {(end - start):.3f}s')
        
        
    def Qvalues(self):
        Qs = np.zeros((self.numstates, self.numactions))
        for state in range(self.numstates):
            Qs[state] = np.sum(self.transitions[state] *\
                               ((1-self.pfail[state]) * (self.rewards[state] + self.Discount * np.tile(self.V, (self.numactions, 1))) +\
                                (np.tile(self.pfail[state], (self.numactions, 1)) * (self.rewards[state]+self.failReward)) ), axis=1)
        print(pd.DataFrame(Qs, columns=['Left', 'Down', 'Right', 'Up']))


    def render(self, safe_path, map=False):
        
        shape = np.array(self.map.shape)
        probs = np.zeros([shape[0],shape[1],2])
        probabilities = self.probability.reshape([shape[0],shape[1],4])
        for i in range(shape[0]):
            for j in range(shape[1]):
                argmax = np.argwhere(probabilities[i,j,:]==1)
                if 0 in argmax:
                    probs[i,j] = np.array([-1,0])
                if 1 in argmax:
                    probs[i,j] = np.array([0,-1])
                if 2 in argmax:
                    probs[i,j] = np.array([1,0])
                if 3 in argmax:
                    probs[i,j] = np.array([0,1])
        
        plt.figure(figsize=shape)
        if map == True:
            sns.heatmap(self.map, cbar=False, cmap='Purples', alpha=0.5, square=True, annot=True, mask=safe_path,
                        linewidths=0.4, linecolor='gray', annot_kws={'color':'black'})
            sns.heatmap(self.map, cbar=False, cmap='tab10', alpha=0.5, square=True, annot=True, mask=1-safe_path,
                        linewidths=0.4, linecolor='gray', annot_kws={'color':'black'})
            plt.title('Map of Environment')
            plt.xticks([])
            plt.yticks([])
        else:
            sns.heatmap(self.map, cbar=False, cmap='tab10', alpha=0.5, square=True, mask=1-safe_path,
                        linewidths=0.4, linecolor='gray')
            sns.heatmap(self.V.reshape(shape), annot=True, cbar=False, fmt='.2f', alpha=0, annot_kws={'color':'black'})
            for x in range(shape[0]):
                for y in range(shape[1]):
                    plt.quiver(y+.5, x+.8, probs[x,y,0], probs[x,y,1], scale=5*shape[0], 
                               width=0.004, headwidth=5, color='#3A4F7A')
            plt.title('Values & Policies')
            plt.xticks([])
            plt.yticks([])

    