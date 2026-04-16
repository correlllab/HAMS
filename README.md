# Humanoid_Simulation

## Important!!!
The IsaacLab component was tested on NVIDIA driver version 580.x, Ubuntu 24.04, a NVIDIA GeForce RTX 4060, AMD Ryzen 9 8945HS w/ Radeon 78. In testing, it did not work with 595 drivers.

## Installation
- Just copy and paste in a terminal:
    ```bash
    cd ~
    #Or wherever you want, just make sure to change paths in docker_run.sh
    git clone https://github.com/correlllab/Humanoid_Simulation.git
    cd Humanoid_Simulation/scripts
    ./docker_build.sh
    #Change paths in docker_run to the ones on your system.
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
## Mini Demo Video(Will be updated soon)
https://github.com/user-attachments/assets/61b75083-ebab-4e3e-9ba5-d7bf31474d01
## Important Notes for Use of Yutong's ROS2 Controller in Sim
If you plan on using Yutong's ROS2 controller for test, it is best to comment out the line that raises the E-Stop exception (which can be found with just a grep). IsaacSim joint states are below the minimum tolerances for E-Stops, so everything will just error out.

