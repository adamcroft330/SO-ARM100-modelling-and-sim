# src/kinematics.py
import numpy as np
from typing import Tuple, List, Union, Optional
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
        
    Raises:
        ValueError: If input parameters contain NaN or infinity values
    """
    # Input validation
    params = [theta, d, a, alpha]
    if any(np.isnan(p) for p in params):
        raise ValueError("DH parameters contain NaN values")
    if any(np.isinf(p) for p in params):
        raise ValueError("DH parameters contain infinity values")
    
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])

def get_joint_positions(joint_angles: List[float], 
                       config: RobotConfig = default_config) -> np.ndarray:
    """Calculate positions of all joints for visualization.
    
    This function returns the positions of all joints including servo centers
    and connecting points for accurate visualization.
    
    Args:
        joint_angles: List of 6 joint angles
        config: Robot configuration object
        
    Returns:
        numpy array: Nx3 array of joint positions and key points
    """
    # Validate joint angles
    valid, message = config.check_joint_limits(joint_angles)
    if not valid:
        raise ValueError(f"Invalid joint angles: {message}")
    
    # Get DH parameters
    dh_params = config.get_dh_parameters(joint_angles)
    
    # Initialize positions array for all key points
    positions = []
    
    # Base position and center
    positions.append(np.array([0, 0, 0]))  # Base bottom center
    positions.append(np.array([0, 0, config.base_height]))  # Base top center
    
    # Calculate transformations and positions
    T = np.eye(4)
    
    # Add positions for each joint's key points
    for i, (theta, d, a, alpha) in enumerate(dh_params):
        # Store current position before transform
        pos = T[:3, 3].copy()
        positions.append(pos)  # Joint base position
        
        # Update transformation
        T = T @ transform_matrix(theta, d, a, alpha)
        
        # Add additional points for visualization if needed
        if i == 1:  # Shoulder joint
            # Add point for upper arm
            pos = T[:3, 3].copy()
            positions.append(pos - np.array([0, 0, config.servo_height/2]))
            
        elif i == 2:  # Elbow joint
            # Add point for forearm
            pos = T[:3, 3].copy()
            positions.append(pos - np.array([0, 0, config.servo_height/2]))
            
        elif i == 4:  # Wrist roll joint
            # Add points for gripper mount
            pos = T[:3, 3].copy()
            positions.append(pos)
    
    # Add final end-effector position
    positions.append(T[:3, 3])
    
    return np.array(positions)

def forward_kinematics(joint_angles: List[float], 
                      config: RobotConfig = default_config,
                      return_all_transforms: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], 
                                                                 Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
    """Calculate end-effector position and orientation given joint angles.
    
    Args:
        joint_angles: List of 6 joint angles
        config: Robot configuration object
        return_all_transforms: If True, return all intermediate transformations
        
    Returns:
        If return_all_transforms is False:
            tuple: (position [x,y,z], orientation [roll,pitch,yaw])
        If return_all_transforms is True:
            tuple: (position [x,y,z], orientation [roll,pitch,yaw], list of transforms)
    """
    # Input validation
    if not isinstance(joint_angles, (list, np.ndarray)) or len(joint_angles) != 6:
        raise ValueError("joint_angles must be a list or array of length 6")
    
    if any(np.isnan(angle) for angle in joint_angles):
        raise ValueError("Joint angles contain NaN values")
        
    if any(np.isinf(angle) for angle in joint_angles):
        raise ValueError("Joint angles contain infinity values")
    
    # Validate joint angles
    valid, message = config.check_joint_limits(joint_angles)
    if not valid:
        raise ValueError(f"Invalid joint angles: {message}")
    
    # Calculate cumulative transformation
    transforms = [np.eye(4)]  # Include initial transform
    T = np.eye(4)
    dh_params = config.get_dh_parameters(joint_angles)
    
    for params in dh_params:
        T = T @ transform_matrix(*params)
        transforms.append(T.copy())
    
    # Extract position and orientation
    position = T[:3, 3]
    
    # Convert rotation matrix to Euler angles (ZYX convention)
    R = T[:3, :3]
    orientation = rotm2euler(R)
    
    if return_all_transforms:
        return position, orientation, transforms
    return position, orientation

def rotm2euler(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Euler angles (ZYX convention).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        numpy array: [roll, pitch, yaw] in radians
    """
    # Handle gimbal lock cases
    if abs(R[2,0]) > 0.9999:
        # Gimbal lock case
        yaw = np.arctan2(R[1,2], R[1,1])
        if R[2,0] > 0:  # pitch = -90°
            pitch = -np.pi/2
            roll = 0
        else:  # pitch = 90°
            pitch = np.pi/2
            roll = 0
    else:
        # Normal case
        pitch = -np.arcsin(R[2,0])
        roll = np.arctan2(R[2,1], R[2,2])
        yaw = np.arctan2(R[1,0], R[0,0])
        
        # Ensure angles are in the correct range
        roll = np.mod(roll + np.pi, 2*np.pi) - np.pi
        pitch = np.mod(pitch + np.pi/2, np.pi) - np.pi/2
        yaw = np.mod(yaw + np.pi, 2*np.pi) - np.pi
    
    return np.array([roll, pitch, yaw])

def calculate_jacobian(joint_angles: List[float], 
                      config: RobotConfig = default_config) -> np.ndarray:
    """Calculate the geometric Jacobian matrix for the 6-DOF robot.
    
    Args:
        joint_angles: List of 6 joint angles
        config: Robot configuration object
        
    Returns:
        numpy array: 6x6 Jacobian matrix (linear and angular velocities)
    """
    # Reuse transformations from forward kinematics
    _, _, transforms = forward_kinematics(joint_angles, config, return_all_transforms=True)
    
    # Initialize Jacobian matrix
    J = np.zeros((6, 6))
    
    # Calculate linear and angular components for each joint
    end_position = transforms[-1][:3, 3]
    
    for i in range(6):
        if i == 0:
            # Base joint - rotation about z-axis
            z_axis = np.array([0, 0, 1])
            position = np.array([0, 0, config.base_height])
        else:
            # Get z-axis of previous transform
            z_axis = transforms[i][:3, :3] @ np.array([0, 0, 1])
            position = transforms[i][:3, 3]
        
        # Linear velocity component
        J[:3, i] = np.cross(z_axis, end_position - position)
        # Angular velocity component
        J[3:, i] = z_axis
    
    return J

def inverse_kinematics(target_position: np.ndarray,
                      target_orientation: Optional[np.ndarray] = None,
                      initial_guess: Optional[List[float]] = None,
                      config: RobotConfig = default_config,
                      max_iterations: int = 100,
                      tolerance: float = 1e-3) -> Tuple[List[float], bool]:
    """Calculate joint angles for a desired end-effector pose using damped least squares.
    
    Args:
        target_position: Desired end-effector position [x,y,z]
        target_orientation: Optional desired orientation [roll,pitch,yaw]
        initial_guess: Initial joint angles guess
        config: Robot configuration object
        max_iterations: Maximum iterations for convergence
        tolerance: Error tolerance for convergence
    
    Returns:
        tuple: (joint_angles, success)
    """
    # Initialize from guess or home position
    current_joints = np.array(initial_guess if initial_guess is not None 
                            else config.home_position)
    
    for iteration in range(max_iterations):
        # Get current end-effector pose
        current_position, current_orientation = forward_kinematics(current_joints, config)
        
        # Calculate position error
        position_error = target_position - current_position
        
        # Calculate orientation error if target orientation is specified
        if target_orientation is not None:
            orientation_error = target_orientation - current_orientation
            # Wrap angle differences to [-pi, pi]
            orientation_error = np.arctan2(np.sin(orientation_error), 
                                         np.cos(orientation_error))
            error = np.concatenate([position_error, orientation_error])
        else:
            error = position_error
            
        # Check convergence
        if np.linalg.norm(error) < tolerance:
            return list(current_joints), True
        
        # Calculate Jacobian
        J = calculate_jacobian(current_joints, config)
        if target_orientation is None:
            J = J[:3, :]  # Use only position rows if no orientation target
        
        # Damped least squares parameters
        lambda_sq = 0.1 * 0.1
        
        # Calculate joint corrections
        J_T = J.T
        correction = J_T @ np.linalg.inv(J @ J_T + lambda_sq * np.eye(len(error))) @ error
        
        # Apply correction with step size
        step_size = 0.5
        current_joints += step_size * correction
        
        # Ensure joint limits
        valid, _ = config.check_joint_limits(current_joints)
        if not valid:
            # Project back to valid range
            for i, (name, angle) in enumerate(zip(['base', 'shoulder', 'elbow', 
                                                 'wrist_pitch', 'wrist_roll', 'gripper'], 
                                                current_joints)):
                min_angle, max_angle = config.joint_limits[name]
                current_joints[i] = np.clip(angle, min_angle, max_angle)
    
    return list(current_joints), False
