# Polar Positioning System - Usage Guide

## Overview

The Polar Positioning System is a ROS2-based drone positioning controller that commands the drone in polar coordinates (radius, theta angle, altitude). It supports both **absolute positioning** (go to a specific location) and **relative velocity commands** (move at specified speeds).

## Quick Start

### 1. Launch the System

#### Option A: Launch main position controller

```bash
ros2 launch polar_system position_system.launch.py
```

#### Option B: Manual node launch

```bash
# Terminal 1: Start the position controller
ros2 run polar_system position_system

# Terminal 2: Provide fake estimated center (for testing)
ros2 run polar_system one_shot_fake_target

# Terminal 3: Send commands via keyboard
ros2 run polar_system keyboard_teleop

# Terminal 4: Monitor RC input (if using controller)
ros2 run polar_system controller_interface
```

---

## System Architecture

### Main Nodes

| Node | Purpose | Input Topics | Output Topics |
|------|---------|--------------|----------------|
| `position_system` | Main controller; computes accelerations | `/mavros/local_position/pose`, `/mavros/local_position/velocity_local`, `/polar/goal_pose`, `/polar/estimated_center`, `/mavros/global_position/compass_hdg` | `/mavros/setpoint_raw/local`, `/polar/reached_target`, `/polar/abort_brake` |
| `keyboard_teleop` | Keyboard input for relative mode commands | - | `/polar/goal_pose` |
| `controller_interface` | RC remote input reader | `/mavros/rc/in` | `/polar/goal_pose`, `/polar/activation` |
| `one_shot_fake_target` | Test target generator | - | `/polar/estimated_center` |

### Topic Namespace

All custom topics use `/polar/` prefix:

```
/polar/goal_pose                    # Target command (TargetPosePolar)
/polar/estimated_center             # Target/object center location (PoseStamped)
/polar/activation                   # Start/stop approach (String)
/polar/controller_activation        # RC controller start/stop (String)
/polar/reached_target               # Target reached feedback (Bool)
/polar/abort_brake                  # Emergency abort signal (String)
```

---

## Operating Modes

### Mode 1: Absolute Positioning (relative=false)

**Purpose:** Drone moves to a specific location in polar coordinates.

**Command Format:**
```python
msg.relative = False
msg.r = 5.0              # Distance from target (meters)
msg.theta = 90.0         # Angle from target (degrees, 0=North, 90=East)
msg.z = -2.0             # Altitude offset from target (meters, negative=below)
msg.relative_theta = False  # Absolute angle mode
```

**Behavior:**
- Drone moves to position (r, theta, z) relative to the target
- Holds position until new command received
- NaN values latch to current position (see **Latching Feature** below)

**Example:**
```
r=5m, theta=0°, z=-2m → Drone 5m North of target, 2m below it
```

### Mode 2: Relative Velocity Mode (relative=true)

**Purpose:** Drone moves at specified velocities in polar coordinates.

**Command Format:**
```python
msg.relative = True
msg.v_r = 0.5            # Radial velocity: positive=inward, negative=outward (m/s)
msg.v_theta = 0.2        # Tangential velocity (m/s)
msg.v_z = -0.1           # Vertical velocity (m/s, negative=down)
msg.yaw_rate = 0.1       # Yaw rotation rate (rad/s)
```

**Behavior:**
- Drone orbits/spirals around target at specified speeds
- Continuous motion until command changes or stops

**Example:**
```
v_r=-0.5, v_theta=0.1 → Drone spirals outward around target
```

### Mode 3: Relative Theta (absolute mode with relative_theta=true)

**Purpose:** Rotate around target by an offset angle.

**Command Format:**
```python
msg.relative = False
msg.relative_theta = True
msg.theta = 10.0         # Offset angle in degrees (CCW positive)
msg.r = 5.0              # Distance (absolute)
msg.z = -2.0             # Altitude (absolute)
```

**Behavior:**
- Moves drone 10° CCW from current angular position
- Each command adds to current position
- Combined with absolute r and z

**Example:**
```
theta=10° → Move 10° CCW from where drone currently is
theta=-20° → Move 20° CW from current position
```

---

## Latching Feature (NaN Commands)

When you send **NaN** for a parameter in **absolute mode**, the system latches to the drone's **current position** for that axis.

**Example Workflow:**

```python
# Step 1: Move to absolute position
msg.relative = False
msg.r = 5.0
msg.theta = 90.0
msg.z = -2.0
pub.publish(msg)
# → Drone moves to (5m East, 2m below target)

# Step 2: Hold radius and altitude, only adjust angle
msg.r = float('nan')     # Latch current radius
msg.theta = 45.0         # New angle
msg.z = float('nan')     # Latch current altitude
pub.publish(msg)
# → Drone holds 5m distance and -2m altitude, rotates to 45°

# Step 3: Freeze everything
msg.r = float('nan')
msg.theta = float('nan')
msg.z = float('nan')
pub.publish(msg)
# → Drone holds current position (no movement)
```

**Use Case:** Gradually adjust approach without re-commanding distance/altitude each time.

---

## Keyboard Teleop Usage

Launch keyboard controller:
```bash
ros2 run polar_system keyboard_teleop
```

### Key Bindings (Relative Mode)

| Key | Action |
|-----|--------|
| `w` / `s` | Increase/decrease radial distance `r += 0.1 / 0.1` |
| `a` / `d` | Increase/decrease tangential velocity `v_theta ± 0.1 m/s` |
| `r` / `f` | Increase/decrease vertical velocity `v_z ± 0.1 m/s` |
| `z` / `x` | Increase/decrease yaw rate `± 0.02 rad/s` |
| `space` | Reset all velocities to zero |
| `h` | Print help |
| `ctrl+c` | Exit |

**Status Display:**
```
r=0.50  v_theta=0.20, v_z=-0.10, yaw_rate = 0.02
```

---

## RC Controller Interface

The `controller_interface` node maps RC transmitter channels to polar commands:

### RC Channel Mapping

| Channel | Function | Output |
|---------|----------|--------|
| 1 | Roll (Aileron) | `v_theta` (tangential velocity) |
| 2 | Pitch (Elevator) | `v_r` (radial velocity, inward=positive) |
| 3 | Throttle | `v_z` (vertical velocity) |
| 4 | Yaw (Rudder) | `yaw_rate` |
| 7 | Activation Switch | Start/stop approach (>1700 PWM = active) |

### Parameters

Set in launch file or via ROS2 params:

```bash
ros2 run polar_system controller_interface \
  --ros-args \
  -p v_r_max:=1.5 \
  -p v_theta_max:=2.0 \
  -p v_z_max:=0.5 \
  -p yaw_max:=0.785
```

---

## Configuration Parameters

### position_system Parameters

Edit or override these in launch files:

```yaml
# Topics
topic_pose: "/mavros/local_position/pose"
topic_vel: "/mavros/local_position/velocity_local"
topic_goal_polar: "/polar/goal_pose"
topic_estimated_center: "/polar/estimated_center"
topic_activation: "/polar/activation"
topic_raw_setpoint: "/mavros/setpoint_raw/local"

# Control rate
control_rate: 25.0  # Hz

# Safety limits
centripetal_limit: 1.5      # Max centripetal acceleration (m/s²)
minimal_margin: 2.0         # Hard keep-out radius (m)
soft_repulsion_initial_radius: 5.0  # Soft zone radius (m)
reach_threshold: 0.2        # Distance to declare "reached" (m)

# Filtering
alpha: 0.1  # Radial velocity filter rate

# Logging
csv_path: "approach_log_polar.csv"
talk: true
log: true
set_msg_interval: true
msg_interval_rate: 25.0

# PID Gains (absolute and relative modes have separate controllers)
pid_r_kp: 3.75
pid_r_ki: 1.0
pid_r_kd: 0.0
# ... (see position_system.py for all PID parameters)
```

---

## Feedback & Monitoring

### 1. Reached Target Callback

When the drone reaches the target (within `reach_threshold`):

```
/polar/reached_target → Bool(True)
```

Monitor in terminal:
```bash
ros2 topic echo /polar/reached_target
```

### 2. CSV Logging

Detailed flight log saved to `approach_log_polar.csv`:

```bash
# Fields logged at 10 Hz:
# radius, set_speed_r, meas_speed_r, acc_cmd_r,
# set_speed_theta, meas_speed_theta, acc_cmd_theta,
# set_speed_z, meas_speed_z, acc_cmd_z,
# acc_x, acc_y, acc_z,
# pid_r_P, pid_r_I, pid_r_D,
# ... (50+ fields total)
```

**Analyze with Python:**
```python
import pandas as pd
df = pd.read_csv('approach_log_polar.csv')
df[['radius', 'set_speed_r', 'meas_speed_r']].plot()
```

### 3. ROS2 Topic Monitoring

Monitor in real-time:

```bash
# See goal commands
ros2 topic echo /polar/goal_pose

# Monitor drone position
ros2 topic echo /mavros/local_position/pose

# Check velocity commands sent
ros2 topic echo /mavros/setpoint_raw/local

# View target reached feedback
ros2 topic echo /polar/reached_target
```

### 4. Info Logger Output

If `talk: true`, the system logs status at 4 Hz:

```
[INFO] Distance : 5.234, vel_r: 0.123
[INFO] yaw offset : 0.045 , total_yaw_err = 0.012
[INFO] Processing time: 0.00234 s
```

---

## Complete Example: Circular Orbit

Send the drone in a 5-meter circular orbit around the target:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from custom_interfaces.msg import TargetPosePolar

class CircleOrbitTest(Node):
    def __init__(self):
        super().__init__('circle_test')
        self.pub = self.create_publisher(TargetPosePolar, '/polar/goal_pose', 10)
        self.timer = self.create_timer(1.0, self.send_command)
    
    def send_command(self):
        msg = TargetPosePolar()
        msg.relative = True          # Velocity mode
        msg.v_r = 0.0                # No radial movement
        msg.v_theta = 0.5            # Tangential velocity (orbits)
        msg.v_z = 0.0                # No vertical movement
        msg.yaw_rate = 0.5           # Rotate drone to face direction
        self.pub.publish(msg)

if __name__ == '__main__':
    rclpy.init()
    node = CircleOrbitTest()
    rclpy.spin(node)
```

---

## Troubleshooting

### Problem: Drone doesn't move

**Causes:**
1. `approach_activation` topic not receiving "start" command
2. No target location published on `/polar/estimated_center`
3. Drone position not updating (check `/mavros/local_position/pose`)

**Solution:**
```bash
# Check activation state
ros2 topic pub /polar/activation std_msgs/String "data: start"

# Verify target location exists
ros2 topic echo /polar/estimated_center

# Verify drone pose is updating
ros2 topic echo /mavros/local_position/pose
```

### Problem: Drone goes to wrong position with NaN

**Cause:** Latch was set when drone was in wrong position, or coordinate frame mismatch

**Solution:**
- Always send valid (non-NaN) values first to establish known position
- Verify drone position in CSV log
- Check that `relative_theta=false` for absolute mode

### Problem: Oscillation/instability

**Cause:** PID gains too aggressive

**Solution:**
```bash
# Lower proportional gain
ros2 param set /position_system pid_r_abs_kp 0.1
ros2 param set /position_system pid_theta_abs_kp 0.1
```

---

## Safety Features

1. **Soft Repulsion Zone:** Drone speed tapers in `soft_repulsion_initial_radius` (default 5m)
2. **Hard Keep-Out:** Drone forced outward if closer than `minimal_margin` (default 2m)
3. **Centripetal Limit:** Maximum centripetal acceleration capped at `centripetal_limit` (default 1.5 m/s²)
4. **Emergency Abort:** Publish on `/polar/abort_brake` to trigger emergency stop

---

## Launch File Example

Create `launch/polar_demo.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Main controller
        Node(
            package='polar_system',
            executable='position_system',
            parameters=[{
                'reach_threshold': 0.2,
                'centripetal_limit': 1.5,
                'talk': True,
                'log': True,
            }],
            output='screen'
        ),
        # Test target
        Node(
            package='polar_system',
            executable='one_shot_fake_target',
            output='screen'
        ),
        # Keyboard control
        Node(
            package='polar_system',
            executable='keyboard_teleop',
            output='screen'
        ),
    ])
```

Launch:
```bash
ros2 launch polar_system polar_demo.launch.py
```

---

## Advanced: Custom Target Provider

Replace `one_shot_fake_target` with your own node:

```python
import rclpy
from geometry_msgs.msg import PoseStamped

rclpy.init()
node = rclpy.create_node('my_target_provider')
pub = node.create_publisher(PoseStamped, '/polar/estimated_center', 10)

msg = PoseStamped()
msg.header.frame_id = 'map'
msg.pose.position.x = 10.0  # Target location in map frame
msg.pose.position.y = 5.0
msg.pose.position.z = 0.5

pub.publish(msg)
print("Target published")
```

The position controller will automatically track this moving target if you update it continuously.

---

## Support & Documentation

For detailed implementation, see:
- [position_system.py](polar_system/position_system.py) - Main controller logic
- [TargetPosePolar.msg](../custom_interfaces/msg/TargetPosePolar.msg) - Message definition
- Parameter files in `config/` folder

