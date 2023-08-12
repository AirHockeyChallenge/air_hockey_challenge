import socketserver

import numpy as np
import torch

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper


class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        mdp = AirHockeyChallengeWrapper("tournament")
        agent = self.server.agent_builder(mdp.env_info, **self.server.config)
        while True:
            data = self.request.recv(1024)
            if data == b'':
                break
            operation = str(data[0:1], "utf-8")
            if operation == "A":
                obs = np.frombuffer(data[1:])
                action = agent.draw_action(obs)
                self.request.sendall(action.tobytes())

            if operation == "R":
                agent.episode_start()
                self.request.sendall(b"r")

            if operation == "P":
                obs = np.frombuffer(data[1:])
                for p in agent.preprocessors:
                    obs = p(obs)
                self.request.sendall(obs.tobytes())


class MyTCPServer(socketserver.TCPServer):
    def __init__(self, server_address, RequestHandlerClass, agent_builder, bind_and_activate=True, **config):
        super(MyTCPServer, self).__init__(server_address, RequestHandlerClass, bind_and_activate)
        self.agent_builder = agent_builder
        self.config = config


def run_tournament_server(agent_builder, host, port, log_dir, n_episodes=10, n_cores=-1,
                          seed=None, generate_score=None, quiet=True, render=False, **config):
    # Keep unused arguments so the arguments in **config are consistent
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    with MyTCPServer((host, port), MyTCPHandler, agent_builder, **config) as server:
        server.serve_forever()
