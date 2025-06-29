# Data Replay System for H1 Robot with Inspire Hands

This system allows you to replay recorded teleoperation data in IsaacGym simulation using the H1 humanoid robot with Inspire hands.

## Files

- `replay_data_test.py` - Main replay script that loads and replays JSON data
- `sample_data.json` - Example data file showing the expected JSON format
- `README_replay.md` - This documentation file

## Requirements

- IsaacGym installed and configured
- H1 robot with Inspire hands URDF files in `assets/h1_inspire/urdf/h1_inspire.urdf`
- Python packages: numpy, json, time, cv2, argparse

## Usage

### Running the Replay

```bash
cd teleop
python replay_data_test.py --data_file sample_data.json --playback_speed 1.0
```

### Command Line Arguments

- `--data_file` (required): Path to the JSON data file containing recorded teleoperation data
- `--playback_speed` (optional): Playback speed multiplier (default: 1.0)
  - Use values > 1.0 for faster playback (e.g., 2.0 for 2x speed)
  - Use values < 1.0 for slower playback (e.g., 0.5 for half speed)

### Example Commands

```bash
# Normal speed replay
python replay_data_test.py --data_file sample_data.json

# Fast replay (2x speed)
python replay_data_test.py --data_file sample_data.json --playback_speed 2.0

# Slow replay (half speed)
python replay_data_test.py --data_file sample_data.json --playback_speed 0.5
```

## Data Format

The JSON data file should contain:

```json
{
    "info": {
        "version": "1.0.0",
        "date": "2025-06-03",
        "author": "unitree",
        "joint_names": {...}
    },
    "text": {
        "goal": "Pick up the red cup on the table.",
        "desc": "Task description",
        "steps": "step1: ... step2: ... step3: ..."
    },
    "data": [
        {
            "idx": 0,
            "states": {
                "left_arm": {"qpos": [7 joint positions]},
                "right_arm": {"qpos": [7 joint positions]},
                "left_hand": {"qpos": [6 joint positions]},
                "right_hand": {"qpos": [6 joint positions]}
            },
            "actions": {
                "left_arm": {"qpos": [7 joint positions]},
                "right_arm": {"qpos": [7 joint positions]},
                "left_hand": {"qpos": [6 joint positions]},
                "right_hand": {"qpos": [6 joint positions]}
            }
        }
    ]
}
```

## Robot Joint Mapping

The system maps JSON data to H1 robot joints as follows:

- **Left Arm** (7 DOF): Indices 13-19 in robot DOF array
- **Left Hand** (6 DOF): Indices 20-25 in robot DOF array  
- **Right Arm** (7 DOF): Indices 32-38 in robot DOF array
- **Right Hand** (6 DOF): Indices 39-44 in robot DOF array

## Scene Setup

The simulation environment includes:

- **Ground plane** - Physics ground
- **Table** - Brown wooden table at height 1.0m
- **Red cup** - Target object positioned on the table
- **H1 Robot** - Humanoid robot with Inspire hands positioned near the table

## Controls

During replay:
- **ESC or close window** - Stop replay and exit
- **Ctrl+C** - Interrupt replay gracefully

## Notes

- The replay uses the recorded joint positions from the `states` field
- Joint positions are applied directly to the robot without physics-based control
- The system prints progress information every 10 frames
- Camera view is positioned to show the robot and table clearly
- The red cup represents the target object mentioned in the task description

## Troubleshooting

1. **"Robot DOF count" message** - Verify the H1 Inspire URDF is loaded correctly
2. **JSON format errors** - Check that your data file matches the expected format
3. **Missing assets** - Ensure the H1 Inspire URDF files are in the correct path
4. **Simulation crashes** - Check joint position limits and values in your data 