# Humanoid_Simulation

IsaacLab Simulation for the H1_2 robot.

## Installation

    ```bash
    cd ~
    git clone https://github.com/correlllab/Humanoid_Simulation.git
    cd Humanoid_Simulation/scripts
    ./docker_build.sh
    ./docker_run.sh
    ```

## Usage
- Within container:

    ```bash
    cd /home/code/h12_sim_scripts
    chmod +x launch.sh
    ./launch.sh
    ```

- I strongly recommend keeping the same container for as long as possible. IsaacSim shader compilation takes forever. To restart a stopped (exited) container:

    ```bash
    sudo docker start -ai h12_sim_container
    #or
    sudo docker exec -it h12_sim_container bash
    ```
