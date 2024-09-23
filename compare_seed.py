import numpy as np
import matplotlib.pyplot as plt

wind_size = 50

seed24691 = np.load('C:\\Users\\Hyein\\Documents\\Github\\epelRL\\_Result\\CSTR_DDPG_seed_24691\\learning_stat_history.npy')
seed713 = np.load('C:\\Users\\Hyein\\Documents\\Github\\epelRL\\_Result\\CSTR_DDPG_seed_713\\learning_stat_history.npy')
seed0 = np.load('C:\\Users\\Hyein\\Documents\\Github\\epelRL\\_Result\\CSTR_DDPG_seed_0\\learning_stat_history.npy')
seed149 = np.load('C:\\Users\\Hyein\\Documents\\Github\\epelRL\\_Result\\CSTR_DDPG_seed_149\\learning_stat_history.npy')
seed7789 = np.load('C:\\Users\\Hyein\\Documents\\Github\\epelRL\\_Result\\CSTR_DDPG_seed_7789\\learning_stat_history.npy')

# Cost, Critic, Actor
data = [seed0, seed713[:, :-1], seed24691[:, :-1], seed149, seed7789]

for idx, d in enumerate(data):
    d_var = [[np.std(d[max(0,i+1-wind_size):i+1, j]) for i in range(len(d))] for j in range(3)]
    d = np.concatenate([d, np.array(d_var).T], axis=1)
    data[idx] = d

plt.figure()
fig, ax = plt.subplots(2, 3, figsize=(12,8))
for i in range(6):
    for d in data:
        ax.flat[i].plot(d[:,i])
    ax.flat[i].grid()
plt.legend(['0', '713', '24691', '149', '7789'])
plt.show()

# I like 50 episode window.
