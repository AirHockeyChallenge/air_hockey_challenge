.. _installation:

============
Installation
============

The codebase for the challenge is in the `Github repo <https://github.com/AirHockeyChallenge/air_hockey_challenge>`_.
To submit your solution, you will have to build up a docker image and upload the image with the scripts we provide.

However, the docker image is not well supported for GUI. We suggest you to develop your agent locally in the ``conda``
or ``virtualenv`` environment and only run the docker image without rendering.

Install the Air Hockey Challenge Locally
----------------------------------------

You first need to clone our github repo

.. code-block:: console

    $ git clone https://github.com/AirHockeyChallenge/air_hockey_challenge.git

You can create a ``conda`` environment and install all dependencies

.. code-block:: console

    $ cd air_hockey_challenge
    $ conda create -y -n challenge python=3.8
    $ conda activate challenge
    $ pip install -r requirements.txt
    $ pip install -e .

To verify that everything works you can run our example hit agent:

.. code-block:: console

    $  python run.py -r -e 3dof-hit --example hit-agent --n_cores 1 --n_episodes 5

.. note::
    Note that we pre-installed the CPU version of torch. The evaluation is also done purely on CPU.
    If you want to use Pytorch with GPU for training, please:

    #. Uninstall torch
    #. Install GPU compatible torch please follow the `pytorch instructions <https://pytorch.org/get-started/locally/>`_.


Build the Docker Image
----------------------

We use Docker to provide portable environment which produces platform invariant results.
Our docker image can be viewed as a snapshot of a linux vm with an already set up python environment.
This environment can be modified to accommodate any requirements needed for the development of your agent.
We can then take the modified image and use it to run your agent on our server.

Prerequisite
~~~~~~~~~~~~

1. Install Docker Engine.

    .. important::
        Linux users need to install the **Docker Engine (Server)** instead of the **Docker Desktop**.

        For macOS and Windows user, please use **Docker Desktop**.

    The installation guide of the
    `Docker Engine <https://docs.docker.com/engine/install/>`_ can be found here.

    .. note::
        If you are using linux, please don't forget to do the `post-install <https://docs.docker.com/engine/install/linux-postinstall/>`_ procedure.

2. Install NVIDIA Container Toolkit.
    Our image is build based on the "nvidia/cuda:11.6.2-base-ubuntu20.04". The installation guide of the
    `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_
    can be found here.

Installation
~~~~~~~~~~~~

We provide two options to setup the docker image

a. Get the public available image from the DockerHub

    .. code-block:: console

        $ docker compose pull challenge

b. Build your image locally

    Build your image from the Dockerfile by running

    .. code-block:: console

        $ docker compose build

    The first build may takes several minutes.

Setting up the UI for Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is setting up the graphics forwarding for docker, which is used to render the simulation.
This can be a bit tedious if you use uncommon hardware because docker is not really designed for this use case.
However you do not need this step to submit your solution via docker.
If you prefer you can develop with a :ref:`local <pip_installation>` installation and just use docker to submit your solutions.

The first step is to check if you have ``xauth`` installed

.. code-block:: console

    $ xauth info

If ``xauth`` is installed you need to create a permission key for the docker image to access the local xserver. First,
you need to create a ".docker.xauth" file

.. code-block:: console

    $ touch /tmp/.docker.xauth

Then, run the following command in your console, this will write the display info into the file

.. code-block:: console

    $ xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge -

This has to be regenerated after every reboot, so we suggest adding these command to your ``~/.bashrc``.

Working with Nvidia GPU
~~~~~~~~~~~~~~~~~~~~~~~

If you have a Nvidia GPU available on your device, you can enable it by renaming the ``docker-compose-nvidia.yml`` file to ``docker-compose.yml``.

Using the Image
^^^^^^^^^^^^^^^^

You can run the docker container:

.. code-block:: console

    $ docker compose run challenge

For our purposes a container is a vm that emulates the images we give it. The terminal attaches to the container
and your ready to run some code.

To verify that everything works you can run our example ``hit-agent``

.. code-block:: console

    $ python run.py --n_cores 1 -e 3dof-hit --example hit-agent --n_episodes 5

If you also set up the UI, you can add the ``-r`` flag to the command which renders the simulator.
To exit the container press ``CTRL + d``, this will detach the terminal and stop the container.

A few tips for docker development:

* The 2023-challenge folder in the container is synced with the your host file system in both directions. You can
  develop your solution locally.

* Make sure all dependencies are installed in the submitted docker image.

* Configure the docker remote interpreter in PyCharm, please `pycharm instructions <https://www.jetbrains.com/help/pycharm/using-docker-compose-as-a-remote-interpreter.html#docker-compose-remote>`_

Installing Custom Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you just want to add pip dependencies you can modify the requirements.txt and rebuild the image.

To add other dependencies you can modify the Dockerfile and rebuild the image.
Add a ``RUN your-install-command`` to the Dockerfile below line 34.
Keep in mind to auto-accept all question than might come up during the install.

For example to add nano to the image the command would be

.. code-block:: docker

    RUN apt-get update && apt-get -y install nano

More detailed documentation on Dockerfiles can be found `here <https://docs.docker.com/engine/reference/builder/>`_.


Whats next?
-----------

With you environment setup you can explore the environments and examples we provide and start developing you own agent in ``air_hockey_agent``.
A good start would be understanding the :ref:`challenge framework <framework>` we provide.

While you are welcome to modify any part of the environment, evaluation etc. please keep in mind that we will restore
all all the ``air_hockey_challenge/*`` files as well as ``run.py`` to their original state for evaluation on the server.
So your agent should be backwards compatible with the original.





