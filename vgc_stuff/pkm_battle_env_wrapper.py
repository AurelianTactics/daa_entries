'''
Wraps PkmBattleEnv into something expected by most RL algorithms
'''

from gymnasium import Env
import numpy as np

class PkmBattleEnvWrapper(Env):

    def __init__(self, PkmBattleEnv, opponent_agent):
        self.env = PkmBattleEnv
        self.opponent_agent = opponent_agent
        self.num_resets = -1
        # to do: figure out reward or index
        self.player_index, self.opponent_index = self._get_player_opp_index()
        # if this is needed then do a reset call here. shouldn't be needed
        #self.current_obs_list = [np.zeros((self.env.observation_space.n,))]
        self.current_obs_list = []


    def step(self, action):
        '''
        Get opponent action then step through env
        '''

        opponent_action = self.opponent_agent.get_action(
            self.current_obs_list[self.opponent_index])

        if self.player_index == 0:
            action_list = [action, opponent_action]
        else:
            action_list = [opponent_action, action]

        self.current_obs_list, reward_list, terminated, truncated, info = self.env.step(action_list)

        obs = self.current_obs_list[self.player_index]

        # get custom reward
        #reward = reward_list[self.player_index]
        reward = self._win_loss_reward(terminated, self.player_index)

        return obs, reward, terminated, truncated, info

    def reset(self):
        ''''''
        self.num_resets += 1
    
        self.current_obs_list, info = self.env.reset()
    
        self.player_index, self.opponent_index = self._get_player_opp_index()

        obs = self.current_obs_list[self.player_index]

        return obs, info

    def render(self, mode='console'):
        ''''''
        self.env.render(mode)

    def _get_player_opp_index(self):
        '''
        Get the player and opponent index
        return 0 or 1 depending on which team you are
        Used for accessing obs indices, reward indices,
        telling who winner is etc
        '''
        player_index = self.num_resets % 2
        opp_index = (player_index + 1) % 2

        return player_index, opp_index
        
    def _win_loss_reward(self, terminated, player_index):
        '''
        Does a reward for winning or losing
        winner is -1 unless a winner has been picked
        '''
        reward = 0.
        if terminated:
            winner = self.env.winner

            if winner == 0 or winner == 1:
                if winner == player_index:
                    reward = 1.
                else:
                    reward = -1.
    
        return reward




