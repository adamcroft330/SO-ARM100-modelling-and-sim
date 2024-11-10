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

    # Linear velocity components
    J1 = np.cross(z0, r1)
    J2 = np.cross(z1, r2)
    J3 = np.cross(z2, r3)

    # Construct the Jacobian matrix
    J = np.zeros((6, 4))
    
    # Linear velocity components
    J[0:3, 0] = J1
    J[0:3, 1] = J2
    J[0:3, 2] = J3
    J[0:3, 3] = 0
    
    # Angular velocity components
    J[3:6, 0] = z0
    J[3:6, 1] = z1
    J[3:6, 2] = z2
    J[3:6, 3] = np.array([0, 0, 1])

    return J

def inverse_kinematics_position_only(target_position: np.ndarray,
                                   initial_guess: List[float] = None,
                                   config: RobotConfig = default_config,
                                   max_iterations: int = 100,
                                   tolerance: float = 1e-3) -> Tuple[List[float], bool]:
    """Calculate joint angles for a desired end-effector position (ignoring orientation).
    
    Args:
        target_position: Desired end-effector position [x, y, z]
        initial_guess: Initial joint angles guess. If None, uses home position
        config: Robot configuration object
        max_iterations: Maximum iterations for convergence
        tolerance: Error tolerance for convergence
    
    Returns:
        Tuple[List[float], bool]: (joint_angles, success)
    """
    # Check if target is within reach
    target_distance = np.linalg.norm(target_position)
    if target_distance > config.max_reach:
        print(f"Target at {target_distance:.3f}m is beyond max reach of {config.max_reach:.3f}m")
        return initial_guess if initial_guess is not None else config.home_position, False
    
    if initial_guess is None:
        current_joints = np.array(config.home_position)
    else:
        current_joints = np.array(initial_guess)
    
    print(f"Starting IK solution search:")
    print(f"Target position: {target_position}")
    print(f"Initial joints: {current_joints}")
    
    for iteration in range(max_iterations):
        # Get current end-effector position
        current_position, _ = forward_kinematics(list(current_joints), config)
        
        # Calculate position error
        error = target_position - current_position
        error_magnitude = np.linalg.norm(error)
        
        if iteration % 10 == 0:  # Print debug info every 10 iterations
            print(f"Iteration {iteration}: Error = {error_magnitude:.6f}")
        
        # Check if we've reached the target within tolerance
        if error_magnitude < tolerance:
            print(f"Solution found in {iteration} iterations!")
            return list(current_joints), True
        
        # Calculate Jacobian
        J = calculate_jacobian(list(current_joints), config)
        J = J[:3, :]  # Only use position rows of Jacobian
        
        # Damped least squares solution
        damping = 0.1
        J_T = J.T
        lambda_sq = damping * damping
        correction = J_T @ np.linalg.inv(J @ J_T + lambda_sq * np.eye(3)) @ error
        
        # Apply correction with step size
        step_size = 0.1
        current_joints += step_size * correction
        
        # Enforce joint limits
        for i, joint_name in enumerate(['shoulder', 'elbow', 'wrist', 'gripper']):
            min_angle, max_angle = config.joint_limits[joint_name]
            current_joints[i] = np.clip(current_joints[i], min_angle, max_angle)
    
    print("Failed to converge within maximum iterations")
    return list(current_joints), False

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
