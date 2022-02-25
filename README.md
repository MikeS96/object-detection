# Duckitown segmentations using Transformers 🤖

This is a repo implementing a visual-servoing agent for lane following and obstable avoidance.

## Phase 0: System update

- 💻 Always make sure your Duckietown Shell is updated to the latest version. See [installation instructions](https://github.com/duckietown/duckietown-shell)

- 💻 Update the shell commands: `dts update`

- 💻 Pull latest containers on your laptop: `dts desktop update`

- 🚙 Clean and update your Duckiebot: `dts duckiebot update ROBOTNAME` (where `ROBOTNAME` is the name of your Duckiebot chosen during the initialization procedure.)

- 🚙 Reboot your Duckiebot.

## Phase 1s: Nvidia-Docker setup (optional)

**Note**: Only do if you have nvidia GPU, otherwise go to the next phase.

- 💻 Set-up nvidia-docker2. See [Installation Nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 💻 Follow [this](https://github.com/NVIDIA/nvidia-container-runtime) to change docker runtime, particularly, run the next command:
    ```
    sudo tee /etc/docker/daemon.json <<EOF                             
    {  "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    EOF
    sudo pkill -SIGHUP dockerd
    ```

## Phase 2s: Docker setup

- 💻 Download the following docker image using `docker pull mikes96/challenge-aido_lf-baseline-duckietown-ml:daffy-amd64`

- 💻 Re-tag docker image doing `docker tag <IMAGE-TAG> duckietown/challenge-aido_lf-baseline-duckietown-ml:daffy-amd64
`
- 🤖 Become a transformer!

## Phase 3s: Play around

Build your code using:

    💻$ `dts exercises build`

You can see the segmentation masks in the simulator by running

    💻$ `dts exercises test --sim` 

Sadly, this repo does not run (yet) in the Jetson Nano, nevertheless, you can use your beefy GPU to speed-up inference by running:

    💻$ `dts exercises test --duckiebot_name mycroft --local` 

Congratulations!, you are now a 🤖