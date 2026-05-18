# Fake Front Target + Polar Commands

This is a copy-paste quickstart to:
1. Start `polar`
2. Start `fake_front_target`
3. Trigger a center point 2 m (or configured distance) in front of the drone
4. Send relative commands to Polar

## 1) Build and source

```bash
cd /home/zenith/aeac-2026
colcon build --packages-select polar_system
source install/setup.bash
```

## 2) Start Polar (terminal A)

```bash
source /home/zenith/aeac-2026/install/setup.bash
ros2 run polar_system polar
```

## 3) Start fake_front_target (terminal B)

```bash
source /home/zenith/aeac-2026/install/setup.bash
ros2 run polar_system fake_front_target 
```

## 4) Activate Polar + Trigger fake front target (terminal C)

This single command activates Polar AND triggers fake_front_target to compute/publish the front point:

```bash
source /home/zenith/aeac-2026/install/setup.bash
ros2 topic pub --once /polar/activation std_msgs/msg/String "{data: start}"
```

Verify center was published:

```bash
ros2 topic echo --once /polar/estimated_center
```

## 5) Send relative commands to Polar (terminal C)

Single command (one-shot):

```bash
ros2 topic pub --once /polar/goal_pose custom_interfaces/msg/TargetPosePolar "{
  relative: false,
  r: 3.0,
  theta: .nan,
  z: .nan,
  v_r: 0.0,
  v_theta: 0.0,
  v_z: 0.0,
  yaw_rate: 0.0
}"
```

Continuous orbit-style relative command (5 Hz):

```bash
ros2 topic pub -r 5 /polar/goal_pose custom_interfaces/msg/TargetPosePolar "{
  relative: true,
  relative_theta: false,
  r: 0.0,
  theta: 0.0,
  z: 0.0,
  v_r: 0.0,
  v_theta: 0.6,
  v_z: 0.0,
  yaw_rate: 0.3
}"
```

## 6) Stop Polar

```bash
ros2 topic pub --once /polar/activation std_msgs/msg/String "{data: stop}"
```

## Optional: if you use bringup launch topic names

Some launch files set different topics (`/approach_activation`, `/goal_pose_polar`, `/estimated_center_location`).
In that case, run fake_front_target with matching output topic:

```bash
ros2 run polar_system fake_front_target --ros-args \
  -p topic_front_point:=/estimated_center_location \
  -p topic_activation:=/approach_activation
```

Then publish activation on the launch-configured topics instead of `/polar/...`.
