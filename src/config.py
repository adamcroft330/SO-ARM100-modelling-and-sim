# src/config.py
from dataclasses import dataclass
from math import pi
from typing import Tuple

@dataclass
class RobotConfig:
    """Configuration class for SO-ARM100 robot.
    
    This class contains all physical parameters and constraints of the robot,
    including link lengths, joint limits, and parallel linkage parameters.
    """
    
    # Physical structure parameters (in meters)
    base_height: float = 0.060
    link1_length: float = 0.100  # Main arm segment length
    link2_length: float = 0.100  # Forearm segment length
    gripper_length: float = 0.050  # Gripper mechanism length
    
    # Joint limits (in radians)
    joint_limits: dict = None
    
    # Parallel linkage parameters
    parallel_link1: float = 0.040  # Upper parallel link length
    parallel_link2: float = 0.040  # Lower parallel link length
    parallel_offset: float = 0.020  # Offset from main joints
    
    def __post_init__(self):
        """Initialize joint limits and home position after dataclass creation."""
        self.joint_limits = {
            'shoulder': (0, pi),           # 0 to 180 degrees
            'elbow': (-pi/2, pi/2),        # -90 to 90 degrees
            'wrist': (-pi/3, pi/3),        # -60 to 60 degrees
            'gripper': (0, pi/4)           # 0 to 45 degrees
        }
        
        # Default home position (in radians)
        self.home_position = [
            pi/2,   # Shoulder vertical
            0.0,    # Elbow neutral
            0.0,    # Wrist centered
            pi/6    # Gripper partially open
        ]
        
        # Calculate workspace limits
        self.max_reach = self.link1_length + self.link2_length + self.gripper_length
        self.min_reach = 0.040  # Minimum reach from base
    
    def check_joint_limits(self, joint_angles: list) -> Tuple[bool, str]:
        """Check if joint angles are within limits.
        
        Args:
            joint_angles: List of 4 joint angles [shoulder, elbow, wrist, gripper]
            
        Returns:
            tuple: (is_valid, message)
        """
        names = ['shoulder', 'elbow', 'wrist', 'gripper']
        for i, (name, angle) in enumerate(zip(names, joint_angles)):
            min_angle, max_angle = self.joint_limits[name]
            if not (min_angle <= angle <= max_angle):
                return False, f"{name} joint angle {angle:.2f} exceeds limits [{min_angle:.2f}, {max_angle:.2f}]"
        return True, "Joint angles within limits"
    
    def get_dh_parameters(self, joint_angles: list) -> list:
        """Get Denavit-Hartenberg parameters for current joint configuration.
        
        Args:
            joint_angles: List of 4 joint angles [shoulder, elbow, wrist, gripper]
            
        Returns:
            list: DH parameters as list of [theta, d, a, alpha] for each joint
        """
        return [
            [joint_angles[0], self.base_height, 0, 0],           # Shoulder
            [joint_angles[1], 0, self.link1_length, 0],          # Elbow
            [joint_angles[2], 0, self.link2_length, 0],          # Wrist
            [joint_angles[3], 0, self.gripper_length, 0]         # Gripper
        ]

# Create a default configuration instance
default_config = RobotConfig()
