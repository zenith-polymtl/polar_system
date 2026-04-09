ros2 topic pub --once /approach_activation std_msgs/msg/String "{data: 'start'}"


ros2 topic pub --once /approach_activation std_msgs/msg/String "{data: 'stop'}"

ros2 topic pub --once /goal_pose_polar custom_interfaces/msg/TargetPosePolar "{
  relative: false,
  r: 7.0,
  theta: .nan,
  z: .nan,
  v_r: 0.0,
  v_theta: 0.0,
  v_z: 0.0,
  yaw_rate: 0.0
}"

/aeac/internal/auto_approach/target_position

ros2 topic pub --once /aeac/internal/auto_approach/target_positioncustom_interfaces/msg/TargetPosePolar "{
  relative: false,
  r: 7.0,
  theta: .nan,
  z: .nan,
  v_r: 0.0,
  v_theta: 0.0,
  v_z: 0.0,
  yaw_rate: 0.0
}"