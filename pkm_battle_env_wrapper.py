'''
Wraps PkmBattleEnv into something expected by most RL algorithms
'''

from gymnasium import Env


class PkmBattleEnvWrapper(Env):

    def __init__(self, PkmBattleEnv):
        self.env = PkmBattleEnv
        # to do: figure out reward or index
        self.obs_reward_index = self._get_obs_reward_index()
        pass


    def step(self, actions):
        obs_list, reward_list, terminated, truncated, info = self.env.step(actions)
        obs = obs_list[self.obs_reward_index]
        reward = reward_list[self.obs_reward_index]

        return obs, reward, terminated, truncated, info


    def reset(self):
        
        obs_list, info = self.env.reset()
        self.obs_reward_index = self._get_obs_reward_index()
        obs = obs_list[self.obs_reward_index]

        return obs, info

    
    def render(self, mode='console'):
        self.env.render(mode)

    def _get_obs_reward_index(self):

        pass
        # return 0 or 1 depending on which team you are
