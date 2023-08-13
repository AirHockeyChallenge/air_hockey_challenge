import socket
import time
import os

import numpy as np

from mushroom_rl.core import Agent


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        return te - ts, result

    return timed


class TournamentAgentWrapper(Agent):
    def draw_action(self, state):
        obs_1, obs_2 = np.split(state, 2)
        time_1, action_1 = self.get_action_1(obs_1)
        time_2, action_2 = self.get_action_2(obs_2)

        return action_1, action_2, time_1, time_2

    def episode_start(self):
        self.episode_start_1()
        self.episode_start_2()

    @property
    def preprocessors(self):
        def _preprocessor(obs):
            obs_1, obs_2 = np.split(obs, 2)
            normalized_obs_1 = self.preprocessor_1(obs_1)
            normalized_obs_2 = self.preprocessor_2(obs_2)
            return np.concatenate([normalized_obs_1, normalized_obs_2])

        return [_preprocessor]

    @timeit
    def get_action_1(self, obs_1):
        raise NotImplementedError

    @timeit
    def get_action_2(self, obs_2):
        raise NotImplementedError

    def episode_start_1(self):
        raise NotImplementedError

    def episode_start_2(self):
        raise NotImplementedError

    def preprocessor_1(self, obs_1):
        return obs_1

    def preprocessor_2(self, obs_2):
        return obs_2


class SimpleTournamentAgentWrapper(TournamentAgentWrapper):
    def __init__(self, env_info, agent_1, agent_2):
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        self.episode_start_1 = self.agent_1.episode_start
        self.episode_start_2 = self.agent_2.episode_start

    @timeit
    def get_action_1(self, obs_1):
        return self.agent_1.draw_action(obs_1)

    @timeit
    def get_action_2(self, obs_2):
        return self.agent_2.draw_action(obs_2)

    def preprocessor_1(self, obs_1):
        for p in self.agent_1.preprocessors:
            obs_1 = p(obs_1)
        return obs_1

    def preprocessor_2(self, obs_2):
        for p in self.agent_2.preprocessors:
            obs_2 = p(obs_2)
        return obs_2


class RemoteTournamentAgentWrapper(TournamentAgentWrapper):
    def __init__(self, ip_1, port_1, ip_2, port_2, container_id_1, container_id_2, name_1, name_2, log_dir):
        self.container_id_1 = container_id_1
        self.container_id_2 = container_id_2

        self.name_1 = name_1
        self.name_2 = name_2
        self.log_dir = log_dir

        # Create a socket (SOCK_STREAM means a TCP socket)
        self.socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_1.connect((ip_1, port_1))

        self.socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_2.connect((ip_2, port_2))

    def episode_start_1(self):
        assert self._get_from_socket(self.socket_1, "R") == b'r'

    def episode_start_2(self):
        assert self._get_from_socket(self.socket_2, "R") == b'r'

    @timeit
    def get_action_1(self, obs_1):
        return np.frombuffer(self._get_from_socket(self.socket_1, "A", obs_1)).reshape(2, -1)

    @timeit
    def get_action_2(self, obs_2):
        return np.frombuffer(self._get_from_socket(self.socket_2, "A", obs_2)).reshape(2, -1)

    def preprocessor_1(self, obs_1):
        return np.frombuffer(self._get_from_socket(self.socket_1, "P", obs_1))

    def preprocessor_2(self, obs_2):
        return np.frombuffer(self._get_from_socket(self.socket_2, "P", obs_2))

    def _get_from_socket(self, socket, operation, obs=None):
        data = bytearray(operation, "utf-8")
        if obs is not None:
            data.extend(obs.tobytes())
        socket.sendall(data)
        return socket.recv(1024)

    def __del__(self):
        import docker

        client = docker.from_env()

        try:
            with open(os.path.join(self.log_dir, f"out_{self.name_1}.log"), 'ab') as f:
                for line in client.api.logs(self.container_id_1, stream=True, follow=False):
                    f.write(line)

            with open(os.path.join(self.log_dir, f"out_{self.name_2}.log"), 'ab') as f:
                for line in client.api.logs(self.container_id_2, stream=True, follow=False):
                    f.write(line)
        except Exception as e:
            print(e)

        client.api.stop(self.container_id_1)
        client.api.stop(self.container_id_2)
