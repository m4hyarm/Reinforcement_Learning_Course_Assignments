from BP_reward import get_reward
import numpy as np

def doctor_A(s_id, drugs):
    drug = np.random.choice(drugs)
    rewards_mean, rewards = [], []
    rwrd = 0
    ref = 0

    for i in range(100):
        reward = get_reward(drug, s_id)
        if reward > ref:
            if drug == 1:
                drug = np.random.choice(drugs, p=[0.8, 0.2])
            else:
                drug = np.random.choice(drugs, p=[0.2, 0.8])

        else:
            if drug == 1:
                drug = np.random.choice(drugs, p=[0.3, 0.7])
            else:
                drug = np.random.choice(drugs, p=[0.7, 0.3])
                
        rwrd += reward
        rewards.append(reward)
        rewards_mean.append(rwrd/(i+1))
        
    return rewards_mean, rewards



def doctor_B(s_id, drugs):
    rewards_mean, rewards = [], []
    rwrd = 0
    
    for i in range(100):
        drug = np.random.choice(drugs)
        reward = get_reward(drug, s_id)
        rwrd += reward
        rewards_mean.append(rwrd/(i+1))
        rewards.append(reward)
        
    return rewards_mean, rewards



def doctor_C(s_id, drugs):
    rewards, drug_1, drug_2 = [], [], []

    i = 0
    for n in range(10):
        reward = get_reward(1, s_id)
        drug_1.append(reward)
        rewards.append(reward)

    for n in range(10):
        reward = get_reward(2, s_id)
        drug_2.append(reward)
        rewards.append(reward)

    i += 20

    while i < 100:

        if max(drug_1) > max(drug_2):
            for n in range(7):
                reward = get_reward(1, s_id)
                drug_1.append(reward)
                rewards.append(reward)
        else:
            for n in range(7):
                reward = get_reward(2, s_id)
                drug_2.append(reward)
                rewards.append(reward)

        for n in range(3):
            drug = np.random.choice(drugs)
            reward = get_reward(drug, s_id)
            if drug == 1:
                drug_1.append(reward)
            else:
                drug_2.append(reward)
            rewards.append(reward)

        i += 10

    rewards_mean = []
    rwrd =0
    for i in range(100):
        rwrd += rewards[i]
        rewards_mean.append(rwrd/(i+1))
        
    return rewards_mean, rewards