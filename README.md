# Humanoid_Simulation
# h1_mujoco

IsaacLab Simulation for the H1_2 robot.

## Installation

- Clone this repo.
- Build docker image:

    ```bash
    cd ~
    git clone https://github.com/correlllab/Humanoid_Simulation.git
    cd Humanoid_Simulation/scripts
    ./docker_build.sh
    ./docker_run.sh
    ```

## Usage

- Run launch.sh script within container:

    ```bash
    cd /home/code/h12_sim_scripts
    chmod +x launch.sh
    ./launch.sh
    ```

