# UR10 Gripper Robot Arm with PPO Training in IsaacLab

This project demonstrates how to **assemble a UR10 robotic arm with a gripper**, integrate it into an **IsaacLab reinforcement learning environment**, and train a **PPO reach policy** that enables the robot to reach target positions. The training setup leverages **IsaacSim**, **IsaacLab**, and **skrl PPO** to simulate and optimize the robot’s behavior across **2000 parallel robots**.

---

## 🚀 Project Overview

- Built a **UR10 + gripper** robot using IsaacSim’s Robot Assembler.  
- Defined robot articulation, joints, and actuators for simulation.  
- Created a **Reach environment** in IsaacLab with configurable scene, actions, observations, rewards, and curriculum.  
- Designed **custom reward functions** for end-effector position and orientation tracking.  
- Trained a **PPO policy** with IsaacLab’s multi-agent setup (`skrl PPO`).  
- Achieved large-scale parallel training with **2000 robots simultaneously**.

---

## 🧰 Prerequisites

- **IsaacSim** and **IsaacLab** installed and working on your machine.
- Access to a **UR10 + gripper USD** (update the path in `ur_gripper.py`).

---

## 🦾 Robot Creation

- Defined the **UR10 with gripper** in [`ur_gripper.py`](./ur_gripper.py).  
- Includes joint definitions, initial poses, actuators for arm and gripper.  
- Example snippet:

```python
UR_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/icepeng/isaac_assets/UR-with-gripper.usd",
        ...
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
        },
    ),
    actuators={...}
)
```

---

## 🏗️ Environment Setup

The environment is defined in [`reach_env_cfg.py`](./reach_env_cfg.py) and registered in [`__init__.py`](./__init__.py).

### Key Components
- **Scene configuration**: Robot articulation, initial pose, velocities.  
- **Environment configuration**: Actions, Observations, Rewards, Terminations, Curriculum, etc.  
- **Gym Registration**:
  - `Template-Reach-v0` (training)
  - `Template-Reach-Play-v0` (inference/playback)

Example registration:

```python
gym.register(
    id="Template-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "reach_env_cfg:ReachEnvCfg",
        "skrl_cfg_entry_point": "agents:skrl_ppo_cfg.yaml",
    },
)
```

---

## ⚙️ PPO Configuration

The [`skrl_ppo_cfg.yaml`](./skrl_ppo_cfg.yaml) contains hyperparameters for the PPO agent, including network architecture, optimizer settings, rollout size, discount factors, and entropy regularization.

---

## ⏱️ Quick Start (with Manager-Based Template)

These steps reflect the exact flow you described.

1. **Create a new IsaacLab project scaffold**
   ```bash
   ./isaaclab.sh
   ```
   - In the **UI**, choose the **Manager-based** option when prompted.

2. **Let the script create the project**
   - Follow the prompts (project name, location, RL options, etc.).  
   - Note the generated **project folder path** (e.g., `source/MyReach` inside your IsaacLab workspace).

3. **Replace this repo’s `Reach` with your scaffolded project**
   - Copy/replace the generated project folder from the **manager_based** location **into this repository**, overwriting the `Reach` project directory (or rename accordingly).
   - Example (adjust paths as needed):
     ```bash
     # Example paths (edit to match your setup)
     ISAACLAB_WS=~/IsaacLab  # where the script generated the project
     NEW_PROJ=$ISAACLAB_WS/source/Reach       # or your chosen name
     THIS_REPO=~/ur10-reach-ppo               # path to this repo

     # Replace the Reach folder in this repo with the newly generated one
     rm -rf "$THIS_REPO/source/Reach"
     mkdir -p "$THIS_REPO/source"
     cp -a "$NEW_PROJ" "$THIS_REPO/source/Reach"
     ```

4. **Install the project in editable mode**
   ```bash
   cd $THIS_REPO
   python -m pip install -e source/Reach
   ```

5. **Update asset paths (if needed)**
   - Edit `ur_gripper.py` to point `usd_path` to your local UR+gripper USD.

6. **Train the PPO policy (example with 2000 envs)**
   ```bash
   python scripts/train.py --task=Template-Reach-v0 --num_envs=2000
   ```

7. **Play with the trained policy**
   ```bash
   python scripts/play.py --task=Template-Reach-Play-v0 --checkpoint=checkpoints/latest.pt
   ```

> 💡 Tip: Keep your **environment IDs** in sync with the Gym registrations in `__init__.py`.

---

## 🎯 Reward Functions

Located in [`rewards.py`](./rewards.py).

- **`position_command_error`**: Penalizes distance between end-effector and commanded position.  
- **`orientation_error`**: Penalizes quaternion mismatch between actual and desired orientation.  
- **`position_command_error_tanh`**: Smooth reward shaping using tanh kernel.  

Example:

```python
def position_command_error(env, command_name, asset_cfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(...)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)
```

---

## 📖 Learning & Notes

- Learned to **assemble robots** with IsaacSim’s Robot Assembler.  
- Created a **scalable RL environment** and trained on **2000 robots simultaneously**.  
- IsaacLab made it straightforward to integrate **skrl PPO** with reinforcement learning backbones.

---

## 🔗 Resources

- Build Your Second Robot on IsaacLab — https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-47+V1&unit=block-v1:DLI+S-OV-47+V1+type@vertical+block@fff0914a40cd4929854106a693f0cd06
