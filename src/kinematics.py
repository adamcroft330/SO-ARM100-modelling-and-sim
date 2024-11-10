# src/kinematics.py
import numpy as np
from typing import Tuple, List, Union
from .config import RobotConfig, default_config

def transform_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Calculate the Denavit-Hartenberg transformation matrix.
    
    Args:
        theta: Joint angle (in radians)
        d: Link offset
        a: Link length
        alpha: Link twist (in radians)
        
    Returns:
        4x4 numpy array transformation matrix
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,             np.sin(alpha),                np.cos(alpha),                d],
        [0,             0,                           0,                             1]
    ])

def forward_kinematics(joint_angles: List[float], 
                      config: RobotConfig = default_config) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate end-effector position and orientation given joint angles.
    
    Args:
        joint_angles: List of 4 joint angles [shoulder, elbow, wrist, gripper]
        config: Robot configuration object
        
    Returns:
        tuple: (position [x,y,z], orientation [roll,pitch,yaw])
    """
    # Validate joint angles
    valid, message = config.check_joint_limits(joint_angles)
    if not valid:
        raise ValueError(f"Joint angles out of limits: {message}")
    
    # Get DH parameters for current configuration
    dh_params = config.get_dh_parameters(joint_angles)
    
    # Initialize transformation matrix
    T = np.eye(4)
    
    # Compute forward kinematics
    for theta, d, a, alpha in dh_params:
        T = T @ transform_matrix(theta, d, a, alpha)
        
    # Extract position
    position = T[:3, 3]
    
    # Extract orientation (roll, pitch, yaw)
    orientation = rotm2euler(T[:3, :3])
    
    return position, orientation

def rotm2euler(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles (ZYX convention).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        numpy array: [roll, pitch, yaw] in radians
    """
    # Handle gimbal lock cases
    if abs(R[2,0]) != 1:
        pitch = -np.arcsin(R[2,0])
        roll = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
        yaw = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))
    else:
        # Gimbal lock case
        yaw = 0
        if R[2,0] == -1:
            pitch = np.pi/2
            roll = yaw + np.arctan2(R[0,1], R[0,2])
        else:
            pitch = -np.pi/2
            roll = -yaw + np.arctan2(-R[0,1], -R[0,2])
    
    return np.array([roll, pitch, yaw])

def calculate_parallel_constraint(elbow_angle: float, 
                                config: RobotConfig = default_config) -> float:
    """Calculate how the parallel linkage affects the elbow joint.
    
    Args:
        elbow_angle: Current elbow joint angle in radians
        config: Robot configuration object
        
    Returns:
        float: Constrained elbow angle in radians
    """
    l1 = config.parallel_link1
    l2 = config.parallel_link2
    offset = config.parallel_offset
    
    # Simplified constraint calculation
    # This is a basic approximation - would need actual geometry for precise calculation
    constrained_angle = elbow_angle * (l1/l2)
    
    return constrained_angle

def get_joint_positions(joint_angles: List[float], 
                       config: RobotConfig = default_config) -> np.ndarray:
    """Calculate positions of all joints for visualization.
    
    Args:
        joint_angles: List of 4 joint angles [shoulder, elbow, wrist, gripper]
        config: Robot configuration object
        
    Returns:
        numpy array: Nx3 array of joint positions including base and end-effector
    """
    # Initialize positions array: base, shoulder, elbow, wrist, gripper
    positions = np.zeros((5, 3))
    
    # Base position
    positions[0] = [0, 0, 0]
    
    # Shoulder position
    positions[1] = [0, 0, config.base_height]
    
    # Calculate cumulative transformations
    T = np.eye(4)
    dh_params = config.get_dh_parameters(joint_angles)
    
    for i, (theta, d, a, alpha) in enumerate(dh_params[:-1], start=2):
        T = T @ transform_matrix(theta, d, a, alpha)
        positions[i] = T[:3, 3]
    
    return positions

def calculate_jacobian(joint_angles: List[float], 
                      config: RobotConfig = default_config) -> np.ndarray:
    """Calculate the Jacobian matrix for the SO-ARM100 robot.
    
    The Jacobian J maps joint velocities to end-effector velocities:
    [ẋ] = J * [θ̇₁]
    [ẏ]     [θ̇₂]
    [ż]     [θ̇₃]
    
    Args:
        joint_angles: List of 4 joint angles [shoulder, elbow, wrist, gripper]
        config: Robot configuration object
        
    Returns:
        numpy array: 6x4 Jacobian matrix (linear and angular velocities)
    """
    theta1, theta2, theta3, theta4 = joint_angles
    l1, l2, l3 = config.link1_length, config.link2_length, config.gripper_length
    
    # Pre-calculate trigonometric terms
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c23 = np.cos(theta2 + theta3)
    s23 = np.sin(theta2 + theta3)

    # Position of end-effector relative to each joint axis
    # These are the moment arms for each joint's contribution
    r1 = np.array([
        l1*c2 + l2*c23 + l3*c23,
        0,
        -l1*s2 - l2*s23 - l3*s23
    ])

    r2 = np.array([
        l2*c23 + l3*c23,
        0,
        -l2*s23 - l3*s23
    ])

    r3 = np.array([
        l3*c23,
        0,
        -l3*s23
    ])

    # Calculate z-axis unit vectors for each joint
    z0 = np.array([0, 1, 0])  # Shoulder rotation axis (y-axis)
    z1 = np.array([0, 0, 1])  # Elbow rotation axis (z-axis)
    z2 = np.array([0, 0, 1])  # Wrist rotation axis (z-axis)

    # Linear velocity components (cross products of rotation axes with moment arms)
    J1 = np.cross(z0, r1)
    J2 = np.cross(z1, r2)
    J3 = np.cross(z2, r3)

    # Construct the Jacobian matrix
    # Top 3 rows: linear velocity components
    # Bottom 3 rows: angular velocity components
    J = np.zeros((6, 4))  # 6x4 matrix (6 DOF output, 4 joint inputs)
    
    # Linear velocity components
    J[0:3, 0] = J1  # Shoulder contribution
    J[0:3, 1] = J2  # Elbow contribution
    J[0:3, 2] = J3  # Wrist contribution
    # Gripper doesn't contribute to linear velocity
    J[0:3, 3] = 0
    
    # Angular velocity components
    J[3:6, 0] = z0  # Shoulder rotation
    J[3:6, 1] = z1  # Elbow rotation
    J[3:6, 2] = z2  # Wrist rotation
    # Gripper rotation (only affects end-effector)
    J[3:6, 3] = np.array([0, 0, 1])

    return J

def check_manipulability(joint_angles: List[float], 
                        config: RobotConfig = default_config) -> float:
    """Calculate the manipulability measure of the current configuration.
    
    Higher values indicate better ability to move in all directions.
    Lower values indicate proximity to singularity.
    
    Args:
        joint_angles: List of 4 joint angles [shoulder, elbow, wrist, gripper]
        config: Robot configuration object
        
    Returns:
        float: Manipulability measure (0 means singularity)
    """
    J = calculate_jacobian(joint_angles, config)
    # Use only linear velocity components (top 3 rows)
    J_linear = J[0:3, 0:3]  # Exclude gripper
    # Calculate manipulability measure
    w = np.sqrt(np.linalg.det(J_linear @ J_linear.T))
    return w

def get_joint_velocities(target_velocity: np.ndarray, 
                        joint_angles: List[float],
                        damping: float = 0.01,
                        config: RobotConfig = default_config) -> np.ndarray:
    """Calculate joint velocities needed for desired end-effector velocity.
    
    Uses damped least squares to handle singularities.
    
    Args:
        target_velocity: Desired end-effector velocity [vx, vy, vz, wx, wy, wz]
        joint_angles: Current joint angles
        damping: Damping factor for singularity robustness
        config: Robot configuration object
        
    Returns:
        numpy array: Joint velocities [shoulder_vel, elbow_vel, wrist_vel, gripper_vel]
    """
    J = calculate_jacobian(joint_angles, config)
    
    # Damped least squares inverse
    J_T = J.T
    lambda_sq = damping * damping
    inv_term = np.linalg.inv(J @ J_T + lambda_sq * np.eye(6))
    J_inv = J_T @ inv_term
    
    # Calculate joint velocities
    joint_velocities = J_inv @ target_velocity
    return joint_velocities

# Add to src/kinematics.py

def inverse_kinematics(target_pose: np.ndarray,
                      initial_guess: List[float] = None,
                      config: RobotConfig = default_config,
                      max_iterations: int = 100,
                      tolerance: float = 1e-3) -> Tuple[List[float], bool]:
    """Calculate joint angles for a desired end-effector pose using iterative method.
    
    Args:
        target_pose: Desired end-effector pose [x, y, z, roll, pitch, yaw]
        initial_guess: Initial joint angles guess. If None, uses home position
        config: Robot configuration object
        max_iterations: Maximum iterations for convergence
        tolerance: Error tolerance for convergence
        
    Returns:
        Tuple[List[float], bool]: (joint_angles, success)
        - joint_angles: [shoulder, elbow, wrist, gripper]
        - success: True if solution found within tolerance
    """
    if initial_guess is None:
        initial_guess = config.home_position
    
    current_joints = np.array(initial_guess)
    
    # Extract position and orientation from target pose
    target_position = target_pose[:3]
    target_orientation = target_pose[3:]
    
    for iteration in range(max_iterations):
        # Get current end-effector pose
        current_position, current_orientation = forward_kinematics(current_joints, config)
        
        # Calculate position and orientation error
        position_error = target_position - current_position
        orientation_error = target_orientation - current_orientation
        
        # Combine errors
        error = np.concatenate([position_error, orientation_error])
        error_magnitude = np.linalg.norm(error)
        
        # Check if we've reached the target within tolerance
        if error_magnitude < tolerance:
            return list(current_joints), True
        
        # Calculate Jacobian
        J = calculate_jacobian(current_joints, config)
        
        # Calculate joint corrections using damped least squares
        damping = 0.01  # Damping factor
        J_T = J.T
        correction = J_T @ np.linalg.inv(J @ J_T + damping**2 * np.eye(6)) @ error
        
        # Apply correction with step size
        step_size = 0.5  # Adjust this for stability/speed trade-off
        current_joints += step_size * correction
        
        # Enforce joint limits
        for i, (joint_name, angle) in enumerate(zip(['shoulder', 'elbow', 'wrist', 'gripper'], 
                                                  current_joints)):
            min_angle, max_angle = config.joint_limits[joint_name]
            current_joints[i] = np.clip(angle, min_angle, max_angle)
    
    # If we get here, we didn't converge
    return list(current_joints), False

def inverse_kinematics_position_only(target_position: np.ndarray,
                                   initial_guess: List[float] = None,
                                   config: RobotConfig = default_config,
                                   max_iterations: int = 100,
                                   tolerance: float = 1e-3) -> Tuple[List[float], bool]:
    """Calculate joint angles for a desired end-effector position (ignoring orientation).
    
    This is often more stable than full pose IK for position-only tasks.
    
    Args:
        target_position: Desired end-effector position [x, y, z]
        initial_guess: Initial joint angles guess. If None, uses home position
        config: Robot configuration object
        max_iterations: Maximum iterations for convergence
        tolerance: Error tolerance for convergence
        
    Returns:
        Tuple[List[float], bool]: (joint_angles, success)
    """
    if initial_guess is None:
        initial_guess = config.home_position
    
    current_joints = np.array(initial_guess)
    
    for iteration in range(max_iterations):
        # Get current end-effector position
        current_position, _ = forward_kinematics(current_joints, config)
        
        # Calculate position error
        error = target_position - current_position
        error_magnitude = np.linalg.norm(error)
        
        # Check if we've reached the target within tolerance
        if error_magnitude < tolerance:
            return list(current_joints), True
        
        # Calculate Jacobian (use only position part)
        J = calculate_jacobian(current_joints, config)[:3, :]  # Only position rows
        
        # Calculate joint corrections using damped least squares
        damping = 0.01
        J_T = J.T
        correction = J_T @ np.linalg.inv(J @ J_T + damping**2 * np.eye(3)) @ error
        
        # Apply correction with step size
        step_size = 0.5
        current_joints += step_size * correction
        
        # Enforce joint limits
        for i, (joint_name, angle) in enumerate(zip(['shoulder', 'elbow', 'wrist', 'gripper'], 
                                                  current_joints)):
            min_angle, max_angle = config.joint_limits[joint_name]
            current_joints[i] = np.clip(angle, min_angle, max_angle)
    
    return list(current_joints), False

# Example usage function
def move_to_target(target_position: np.ndarray,
                  config: RobotConfig = default_config) -> None:
    """Example of how to use inverse kinematics to move to a target position.
    
    Args:
        target_position: Desired [x, y, z] position
        config: Robot configuration object
    """
    # Try to find joint angles for target position
    joint_angles, success = inverse_kinematics_position_only(
        target_position,
        config=config
    )
    
    if success:
        print(f"Solution found: {[f'{angle:.2f}' for angle in joint_angles]}")
        # Here you would send these angles to your actual robot
        
        # Verify solution
        actual_position, _ = forward_kinematics(joint_angles, config)
        error = np.linalg.norm(actual_position - target_position)
        print(f"Position error: {error:.4f} meters")
    else:
        print("Failed to find solution - target may be unreachable")
