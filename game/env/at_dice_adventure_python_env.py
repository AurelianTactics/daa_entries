from game.at_dice_adventure import ATDiceAdventure
import game.env.unity_socket as unity_socket
from gymnasium import Env
from json import loads
import numpy as np

'''
To do

new reward
new done
new reset
basic obs

Elsewhere
action handling
'''

class ATDiceAdventurePythonEnv(Env):
    """
    Implements a custom gyn environment for the Dice Adventure game.

    Modified by https://github.com/AurelianTactics
    """
    def __init__(self,
                 player="Dwarf",
                 id_=0,
                 train_mode=False,
                 server="local",
                 state_version="full",#state_version="character",
                 **kwargs):
        """
        Init function for Dice Adventure gym environment.
        :param player:      (string) The player that will be used to play the game.
        :param id_:         (int) An optional ID parameter to distinguish this environment from others.
        :param train_mode:  (bool) A helper parameter to switch between training mode and play mode. When we test agents,
                                   we will use a "play" mode, where the step function simply takes an action and returns
                                   the next state.
        :param server:      (string) Determines which game version to use. Can be one of {local, unity}.
        :param kwargs:      (dict) Additional keyword arguments to pass into Dice Adventure game. Only applies when
                                   'server' is 'local'.
        """
        self.config = loads(open("game/config/main_config.json", "r").read())
        self.player = player
        self.id = id_
        self.kwargs = kwargs

        ##################
        # STATE SETTINGS #
        ##################
        self.state_version = state_version
        self.mask_radii = {"Dwarf": self.config["OBJECT_INFO"]["OBJECT_CODES"]["1S"]["SIGHT_RANGE"],
                           "Giant": self.config["OBJECT_INFO"]["OBJECT_CODES"]["2S"]["SIGHT_RANGE"],
                           "Human": self.config["OBJECT_INFO"]["OBJECT_CODES"]["3S"]["SIGHT_RANGE"]}

        ##################
        # TRAIN SETTINGS #
        ##################
        self.train_mode = train_mode

        ###################
        # SERVER SETTINGS #
        ###################
        self.server = server
        self.unity_socket_url = self.config["GYM_ENVIRONMENT"]["UNITY"]["URL"]
        self.game = None

        if self.server == "local":
            self.game = ATDiceAdventure(**self.kwargs)

        # AT added things
        self.subgoal_count_current = 0
        self.subgoal_count_prior = 0
        self.subgoal_reward = .333

    def step(self, action):
        """
        Applies the given action to the game. Determines the next observation and reward,
        whether the training should terminate, whether training should be truncated, and
        additional info.
        :param action:  (string) The action produced by the agent
        :return:        (dict, float, bool, bool, dict) See description
        """
        game_state = self.execute_action(self.player, action)

        terminated = self._get_terminated()
        truncated = self._get_truncated()
        reward = self._get_reward(terminated, truncated)

        if terminated or truncated:
            new_obs, info = self.reset()
        else:
            new_obs = self.get_state()
            info = game_state

        return new_obs, reward, terminated, truncated, info

    def close(self):
        """
        close() function from standard gym environment. Not implemented.
        :return: N/A
        """
        pass

    def render(self, mode='console'):
        """
        Prints the current board state of the game. Only applies when `self.server` is 'local'.
        :param mode: (string) Determines the mode to use (not used)
        :return: N/A
        """
        if self.server == "local":
            self.game.render()

    def reset(self, **kwargs):
        """
        Resets the game. Only applies when `self.server` is 'local'.
        :param kwargs:  (dict) Additional arguments to pass into local game server
        :return:        (dict, dict) The initial state when the game is reset, An empty 'info' dict
        """
        self.subgoal_count_prior = 0
        self.subgoal_count_current = 0

        if self.server == "local":
            self.game = ATDiceAdventure(**self.kwargs)
            self.game.reset_board_and_level()
        game_state = self.get_state()
        # to do
        obs = {}
        return obs, game_state

    def execute_action(self, player, game_action):
        """
        Executes the given action for the given player.
        :param player:      (string) The player that should take the action
        :param game_action: (string) The action to take
        :return:            (dict) The resulting state after taking the given action
        """
        if self.server == "local":
            self.game.execute_action(player, game_action)
            next_state = self.get_state()
        else:
            url = self.unity_socket_url.format(player.lower())
            next_state = unity_socket.execute_action(url, game_action)
        return next_state

    def get_state(self, player=None, version=None, server=None):
        """
        Gets the current state of the game.
        :param player: (string) The player whose perspective will be used to collect the state. Can be one of
                                {Dwarf, Giant, Human}.
        :param version: (string) The level of visibility. Can be one of {full, player, fow}
        :param server: (string) Determines whether to get state from Python version or unity version of game. Can be
                                one of {local, unity}.
        :return: (dict) The state of the game

        The state is always given from the perspective of a player and defines how much of the level the
        player can currently "see". The following state version options define how much information this function
        returns.
        - [full]:   Returns all objects and player stats. This ignores the 'player' parameter.

        - [player]: Returns all objects in the current sight range of the player. Limited information is provided about
                    other players present in the state.

        - [fow]:    Stands for Fog of War. In the Unity version of the game, you can see a visibility mask for each
                    character. Black positions have not been observed. Gray positions have been observed but are not
                    currently in the player's view. This option returns all objects in the current sight range (view) of
                    the player plus objects in positions that the player has seen before. Note that any object that can
                    move (such as monsters and other players) are only returned when they are in the player's current
                    view.
        """
        version = version if version else self.state_version
        player = player if player else self.player
        server = server if server else self.server

        if server == "local":
            state = self.game.get_state(player, version)
        else:
            url = self.unity_socket_url.format(player)
            state = unity_socket.get_state(url, version)

        return state


    def _get_terminated(self):
        '''
        Calculates if env is terminated
        '''
        terminated = self.game.terminated

        return terminated

    def _get_truncated(self):
        '''
        Calculated if env is truncated
        '''
        truncated = self.game.is_truncated

        return truncated
    
    def _get_subgoal_reward(self):
        '''
        See if any subgoals reached and reward accordingly

        Returns
        -------
        reward: float
        '''
        self.subgoal_count_current = self.game.get_subgoal_count()

        reward = (self.subgoal_count_current - self.subgoal_count_prior) * self.subgoal_reward

        self.subgoal_count_prior = self.subgoal_count_current

        return reward

    def _get_reward(self, terminated, truncated):
        """
        Calculates the reward the agent should receive based on action taken.

        
        Returns
        -------
        reward: float
        """
        reward = self._get_subgoal_reward()

        if terminated or truncated:
            if self.game.is_tower_reached:
                reward += 1.0
        else:
            reward += 0.

        return reward
    
    def _get_minimum_testing_obs(self, player):
        '''
        One layer for player
        One layer for shrines
        One layer for towers
        '''
        # to do:
        # on reset can init the board basics once based on level
        # then updated here
        # actually maybe this will be standardized accross all levels
        # see how initial training does
        depth = 3
        width = self.game.board.width
        height = self.game.board.height

        obs = np.zeros([width, height, depth], dtype=np.float32)

        player_layer = 0
        shrine_layer = 1
        tower_layer = 2

        player_value = 1.
        shrine_value = 1.
        tower_value = 1.

        obs[self.game.board.objects[self.game.player_code_mapping[player]].x,
            self.game.board.objects[self.game.player_code_mapping[player]].y,
            player_layer] = player_value
        
        for i in range(1, 4):
            goal_code = f'{i}G'
            obs[self.game.board.objects[goal_code].x,
                self.game.board.objects[goal_code].y,
                shrine_layer] = shrine_value
        
        obs[self.game.board.objects[self.game.tower].x,
            self.game.board.objects[self.game.tower].y,
            tower_layer] = tower_value
        
        return obs
