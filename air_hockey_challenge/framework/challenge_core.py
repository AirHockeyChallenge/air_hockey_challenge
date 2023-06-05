import time

import numpy as np

from mushroom_rl.core import Core


class ChallengeCore(Core):
    """
    Wrapper for mushrooms Core. Used to time the agents draw_action function and select indices of actions.
    """
    def __init__(self, *args, action_idx=None, is_tournament=False, time_limit=None, init_state=None, **kwargs):
        """
        Constructor.

        Args:
            action_idx(list, None): Indices of action which should get used. Default is the first n indices where n is
                the action length
            is_tournament(bool, False): Flag that is set when environment is tournament. Needed because the tournament
                Agent keeps track of the time
            time_limit(float, None): Time limit for tournament environment. If draw_action took limit than time_limit
                the previous action is reused
            init_state(list, None): Initial state of the robot. Used as initial value for previous action
        """
        super().__init__(*args, **kwargs)

        if action_idx:
            self.action_idx = action_idx
        else:
            if is_tournament:
                self.action_idx = (np.arange(self.mdp.base_env.action_shape[0][0]), np.arange(self.mdp.base_env.action_shape[1][0]))
            else:
                self.action_idx = np.arange(self.mdp.base_env.action_shape[0][0])

        self.is_tournament = is_tournament
        self.time_limit = time_limit

        self.prev_action = None
        self.init_state = init_state

    def _step(self, render):
        """
        Single step.

        Args:
            render (bool):
                whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the
            agent, the reward obtained, the reached state, the absorbing flag
            of the reached state and the last step flag.

        """
        if self.is_tournament:
            action_1, action_2, time_1, time_2 = self.agent.draw_action(self._state)

            if self.time_limit:
                if time_1 > self.time_limit:
                    action_1 = self.prev_action[0]
                    action_1[1] = 0

                if time_2 > self.time_limit:
                    action_2 = self.prev_action[1]
                    action_2[1] = 0

            self.prev_action = [action_1.copy(), action_2.copy()]
            action = (action_1, action_2)
            duration = [time_1, time_2]

            next_state, reward, absorbing, step_info = self.mdp.step(
                (action[0][self.action_idx[0]], action[1][self.action_idx[1]]))

        else:
            start_time = time.time()
            action = self.agent.draw_action(self._state)
            end_time = time.time()
            duration = (end_time - start_time)

            # If there is an index error here either the action shape does not match the interpolation type or
            # the custom action_idx is wrong
            next_state, reward, absorbing, step_info = self.mdp.step(action[self.action_idx])

        step_info["computation_time"] = duration
        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = not (
                self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return (state, action, reward, next_state, absorbing, last), step_info

    def reset(self, initial_states=None):
        super().reset(initial_states)
        self.prev_action = (np.vstack([self.init_state, np.zeros_like(self.init_state)]),
                            np.vstack([self.init_state, np.zeros_like(self.init_state)]))
