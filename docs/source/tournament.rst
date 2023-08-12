.. _tournament:

Tournament
==========

In this stage, you will develop a whole game playing agent to compete with the agent from other participants.

Installation
------------

1. If you have already cloned the repo, you can pull the ``main`` or ``tournament`` branch from the `Github repo <https://github.com/AirHockeyChallenge/air_hockey_challenge>`_.

2. If you have not cloned the repo. Please finish the :ref:`Installation <installation>` instruction.

Launch Tournament Locally
~~~~~~~~~~~~~~~~~~~~~~~~~

You can run a game of two baseline agents playing against each other.

.. code-block:: console

    $ python run.py -e tournament --example baseline -n 1 -r

The default opponent will always be the Baseline Agent. So to launch a game against it

.. code-block:: console

    $ python run.py -e tournament -n 1 -r

To change the opponent to an arbitrary agent you can use the following script

.. literalinclude:: examples/launch_tournament.py


Evaluation
----------

The competition between two agents will be evaluated through a TCP communication between each docker image.
A ``RemoteTournamentAgentWrapper`` distribute observation and collect the action from each agent through
TCP/IP communication.

To test if the docker communication works properly with your agent. You can run

.. code-block:: console

    $ python scripts/docker_tournament_test.py

This script will build a docker image of the current codebase and start a tournament with this image.

Preparation Phase
-----------------

1. The preparation phase begins on **August 14th** and ends on **October 6th**.

2. During the preparation phase, you can submit your solution to the server and evaluate it with the baseline agent.

3. In this phase, we organize a **Friendship Game** for participants. Participants can set the ``friendship_game: True`` in ``agent_config.yml``. We will automatically set a game if two participants are available. A video demo will be published on the website. **No dataset for friendship games will be available**.


Rules
-----

1. Each game will last 15 minutes (45000 steps). Every 500 steps is treat as one episode. The whole game contains 90 episodes.

2. At the beginning of the game, the puck is initialized at a randomly at one side of the table.

3. If the puck is stays on the side of ``Agent-A`` for more than 15s, i.e., :math:`x_{puck} \leq 1.36`, ``Agent-A`` will collect one **FOUL**. The agent will lose one score for every three **FOULS**. The puck will be initialize randomly on the side of ``Agent-B`` where the agent does not collect **FOUL**.

4. If the puck is stuck in the middle (not reachable by both robots), i.e., :math:`1.36 < x_{puck} < 1.66` and :math:`|v_{x,puck}| < 0.05`, the puck will be reset.

5. If ``Agent-A`` scores a goal the puck will be initialized on the side of ``Agent-B`` in the next round.

6. The deployability score is evaluated per episode, i.e., every 500 steps. The total deployability score is the sum of the episodes.

7. The agent wins the game if the agent collects more points and the deployable score are less than 1.5 times the total number of episodes (i.e. 1.5 * 45000 / 500 = 135).
