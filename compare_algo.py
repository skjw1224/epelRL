import numpy as np
import matplotlib.pyplot as plt

wind_size = 50
algo_name = ['A2C', 'DDPG', 'PPO', 'REPS', 'TD3']
# TRPO: 너무 경향성을 벗어나고 발산하는 모델이라서 제외
# TD3: Critic이 두개라서 평균 취한 뒤에 std 계산
data = [np.load('C:\\Users\\Hyein\\Documents\\Github\\epelRL\\_Result\\CSTR_' + algo +
                '_seed_0\\learning_stat_history.npy') for algo in algo_name]

for idx, d in enumerate(data):
    if algo_name[idx] == 'TD3':
        d = np.vstack([d[:,0], (d[:,1]+d[:,2])/2, d[:,3]]).transpose()
    d_var = [[np.std(d[max(0,i+1-wind_size):i+1, j]) for i in range(len(d))] for j in range(3)]
    d = np.concatenate([d, np.array(d_var).T], axis=1)
    data[idx] = d

plt.figure()
fig, ax = plt.subplots(2, 3, figsize=(12,8))
for i in range(6):
    for d in data:
        ax.flat[i].plot(d[:,i])
    ax.flat[i].grid()
ax.flat[1].set_ylim([-0.00001, 0.00025])
ax.flat[4].set_ylim([-0.00001, 0.0002])
plt.legend(algo_name)
plt.show()

# I like 50 episode window.
