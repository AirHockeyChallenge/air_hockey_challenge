.. _submission:

Solution Submission
===================

Before you can submit your solution, you should finish the `Registration <https://air-hockey-challenge.robot-learning.net/participation>`_ process.
After we validate your information, you will receive a ``team_info.yml`` file in the format:

.. code-block:: yaml

    team_name: "Your Team Name"
    AK: "Your Access Key ID"
    SK: "Your Secret Access Key"
    swr_server: "SWR Server"
    login_key: "Your Login Key"

Replace this ``team_info.yml`` with the ``air_hockey_agent/team_info.yml``.

Before submitting to the server, please first validate if your image works locally.
You can run the script outside the docker container.

.. code-block:: console

    $ scripts/local_test.sh

.. important::
    Please make sure you have the ``air_hockey_agent/agent_builder.py`` implemented.

    For testing purpose, you can copy our :ref:`Dummy Agent <dummy_agent>` example.

.. note::

    The full evaluation will run 1000 episodes. You could change ``n_episodes`` in
    ``air_hockey_agent/agent_config.yml`` to 100 to reduce the number of testing episodes.

It will first build your docker image and takes several minutes. You will see output similar like this

.. code-block:: console

    [+] Building 5.0s (15/15) FINISHED
     => [internal] load .dockerignore                                                                                                                                                                     0.0s
     => => transferring context: 2B                                                                                                                                                                       0.0s
     => [internal] load build definition from Dockerfile                                                                                                                                                  0.0s
     => => transferring dockerfile: 1.25kB                                                                                                                                                                0.0s
     => [internal] load metadata for docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04                                                                                                                        0.0s
     => [internal] load build context                                                                                                                                                                     0.1s
     => => transferring context: 50.33kB                                                                                                                                                                  0.0s
     => [base 1/2] FROM docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04                                                                                                                                     0.0s
     => CACHED [base 2/2] RUN apt-get update && apt-get install -y python3-pip python-is-python3 git                                                                                                      0.0s
     => CACHED [pip-build 1/3] WORKDIR /wheels                                                                                                                                                            0.0s
     => CACHED [pip-build 2/3] COPY requirements.txt .                                                                                                                                                    0.0s
     => CACHED [pip-build 3/3] RUN pip install -U pip      && pip wheel -r requirements.txt                                                                                                               0.0s
     => CACHED [eval 1/5] COPY --from=pip-build /wheels /wheels                                                                                                                                           0.0s
     => CACHED [eval 2/5] WORKDIR /src                                                                                                                                                                    0.0s
     => CACHED [eval 3/5] RUN apt-get update && apt-get -y install ffmpeg libsm6 libxext6 &&     rm -rf /var/cache/apt/* /var/lib/apt/lists/*                                                             0.0s
     => CACHED [eval 4/5] RUN pip install -U pip      && pip install --no-cache-dir     --no-index     -r /wheels/requirements.txt     -f /wheels     && rm -rf /wheels                                   0.0s
     => [eval 5/5] COPY . 2023-challenge/                                                                                                                                                                 3.1s
     => exporting to image                                                                                                                                                                                1.7s
     => => exporting layers                                                                                                                                                                               1.7s
     => => writing image sha256:fb50bd8114ee1c0b7a618a1ff52396a17c8238b50967faf29b857fc42b6a2d1f                                                                                                          0.0s
     => => naming to swr.eu-west-101.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-test                                                                                                         0.0s
    Remove Stopped Containers
    9138dbb09e95
    Remove Dangling Images
    Deleted: sha256:dfd824b63044039fc6aed1cc2fd22addf3251d1d129ef73aa8356a4e2e235d1e
    Start Local Testing
      1%|          | 1/125 [00:06<14:25,  6.98s/it]


If the evaluation starts without error, you will be able to submit your solution:

.. code-block:: bash

    scripts/submit_solution.sh


