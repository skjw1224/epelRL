import abc
import torch
import numpy as np


class BaseBuffer(object, metaclass=abc.ABCMeta):
    def __init__(self, config):
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.s_dim = config['s_dim']
        self.a_dim = config['a_dim']
        self.p_dim = config['p_dim']
        self.device = config['device']
        self.need_derivs = config['need_derivs']
        self.need_deriv_inverse = config['need_deriv_inverse']
        self.need_noise_derivs = config['need_noise_derivs']
        self.is_discrete_action = config['is_discrete_action']

        self.reset()

    def __len__(self):
        return self.current_size

    def reset(self):
        self.current_size = 0
        self.index = 0

        self.states = np.zeros((self.buffer_size, self.s_dim))
        if self.is_discrete_action:
            self.actions = np.zeros((self.buffer_size, 1))
        else:
            self.actions = np.zeros((self.buffer_size, self.a_dim))
        self.rewards = np.zeros((self.buffer_size, 1))
        self.next_states = np.zeros((self.buffer_size, self.s_dim))
        self.dones = np.zeros((self.buffer_size, 1))
        if self.need_derivs:
            self.dfdx = np.zeros((self.buffer_size, self.s_dim, self.s_dim))
            self.dfdu = np.zeros((self.buffer_size, self.s_dim, self.a_dim))
            self.dcdx = np.zeros((self.buffer_size, self.s_dim, 1))
            self.dcdu = np.zeros((self.buffer_size, self.a_dim, 1))
            self.d2cdx2 = np.zeros((self.buffer_size, self.s_dim, self.s_dim))
            self.d2cdxdu = np.zeros((self.buffer_size, self.s_dim, self.a_dim))
            self.d2cdu2 = np.zeros((self.buffer_size, self.a_dim, self.a_dim))
            if self.need_deriv_inverse:
                self.d2cdu2_inv = np.zeros((self.buffer_size, self.a_dim, self.a_dim))
            if self.need_noise_derivs:
                self.Fc = np.zeros((self.buffer_size, self.s_dim, self.p_dim))
                self.dFcdx = np.zeros((self.buffer_size, self.p_dim, self.s_dim, self.s_dim))
                self.dFcdu = np.zeros((self.buffer_size, self.p_dim, self.s_dim, self.a_dim))

    def add(self, experience):
        state, action, reward, next_state, done, deriv = experience

        self.states[self.index] = state.reshape(-1, )
        self.actions[self.index] = action.reshape(-1, )
        self.rewards[self.index] = reward.reshape(-1, )
        self.next_states[self.index] = next_state.reshape(-1, )
        self.dones[self.index] = done

        if self.need_derivs:
            self.dfdx[self.index] = deriv[0]
            self.dfdu[self.index] = deriv[1]
            self.dcdx[self.index] = deriv[2]
            self.dcdu[self.index] = deriv[3]
            self.d2cdx2[self.index] = deriv[4]
            self.d2cdxdu[self.index] = deriv[5]
            self.d2cdu2[self.index] = deriv[6]
            if self.need_deriv_inverse:
                self.d2cdu2_inv[self.index] = deriv[7]
            if self.need_noise_derivs:
                self.Fc[self.index] = deriv[8]
                self.dFcdx[self.index] = deriv[9]
                self.dFcdu[self.index] = deriv[10]
        
        if self.current_size < self.buffer_size:
            self.current_size += 1
        self.index = (self.index + 1) % self.buffer_size

    @abc.abstractmethod
    def sample(self):
        pass

    def to_torch(self, array):
        return torch.tensor(array, dtype=torch.float32, device=self.device)


class ReplayBuffer(BaseBuffer):
    def __init__(self, config):
        BaseBuffer.__init__(self, config)

    def sample(self, use_tensor=True):
        batch_size = min(len(self), self.batch_size)
        indices = np.random.randint(len(self), size=batch_size)

        states = self.states[indices]
        actions= self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        
        if self.need_derivs:
            dfdx = self.dfdx[indices]
            dfdu = self.dfdu[indices]
            dcdx = self.dcdx[indices]
            dcdu = self.dcdu[indices]
            d2cdx2 = self.d2cdx2[indices]
            d2cdxdu = self.d2cdxdu[indices]
            d2cdu2 = self.d2cdu2[indices]
            if self.need_deriv_inverse:
                d2cdu2_inv = self.d2cdu2_inv[indices]
                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2, d2cdu2_inv]
            elif self.need_noise_derivs:
                Fc = self.Fc[indices]
                dFcdx = self.dFcdx[indices]
                dFcdu = self.dFcdu[indices]
                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2, Fc, dFcdx, dFcdu]
            else:
                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2]
        else:
            derivs = None
        
        if use_tensor:
            states = self.to_torch(states)
            actions = self.to_torch(actions)
            rewards = self.to_torch(rewards)
            next_states = self.to_torch(next_states)
            dones = self.to_torch(dones)

            if self.need_derivs:
                derivs = [self.to_torch(deriv) for deriv in derivs]
        
        sample = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'derivs': derivs
        }
        return sample


class RolloutBuffer(BaseBuffer):
    def __init__(self, config):
        BaseBuffer.__init__(self, config)

    def sample(self, use_tensor=True):
        states = self.states
        actions= self.actions
        rewards = self.rewards
        next_states = self.next_states
        dones = self.dones
        
        if self.need_derivs:
            dfdx = self.dfdx
            dfdu = self.dfdu
            dcdx = self.dcdx
            dcdu = self.dcdu
            d2cdx2 = self.d2cdx2
            d2cdxdu = self.d2cdxdu
            d2cdu2 = self.d2cdu2
            if self.need_deriv_inverse:
                d2cdu2_inv = self.d2cdu2_inv
                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2, d2cdu2_inv]
            elif self.need_noise_derivs:
                Fc = self.Fc
                dFcdx = self.dFcdx
                dFcdu = self.dFcdu
                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2, Fc, dFcdx, dFcdu]
            else:
                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2]
        else:
            derivs = None
        
        if use_tensor:
            states = self.to_torch(states)
            actions = self.to_torch(actions)
            rewards = self.to_torch(rewards)
            next_states = self.to_torch(next_states)
            dones = self.to_torch(dones)

            if self.need_derivs:
                derivs = [self.to_torch(deriv) for deriv in derivs]
        
        sample = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'derivs': derivs
        }
        return sample

