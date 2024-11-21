# src/config.py
from dataclasses import dataclass
from math import pi
from typing import Tuple, Dict

@dataclass
class RobotConfig:
    """Configuration class for SO-ARM100 robot.
    
    This class contains all physical parameters and constraints of the robot,
    including link lengths, joint limits, and servo dimensions.
    The configuration is based on the standard 6-DOF design with a spherical
    wrist and perpendicular gripper mount.
    """
    
    # Physical structure parameters (in meters)
    base_height: float = 0.020  # Height of base platform (d₁)
    base_width: float = 0.080   # Width of base platform
    
    # Main linkage lengths (in meters)
    link1_length: float = 0.040  # Upper arm length (a₂)
    link2_length: float = 0.040  # Forearm length (a₃)
    wrist_length: float = 0.020  # Wrist extension length (a₄)
    wrist_offset: float = 0.006  # Wrist to gripper offset (a₅)
    gripper_length: float = 0.024  # Gripper servo length (a₆)
    
    # Servo dimensions (in meters)
    servo_width: float = 0.012    # Standard servo width
    servo_height: float = 0.012   # Standard servo height
    servo_length: float = 0.018   # Standard servo length
    
    # Special dimensions for gripper servo
    gripper_servo_width: float = 0.012   # Gripper servo width
    gripper_servo_height: float = 0.012  # Gripper servo height
    gripper_servo_length: float = 0.024  # Gripper servo length (longer axis)
    
    def __post_init__(self):
        """Initialize joint limits, home position, and DH parameters after dataclass creation."""
        # Joint limits (in radians)
        self.joint_limits = {
            'base': (0, pi),           # S1: 0° to 180°
            'shoulder': (-pi/2, pi/2),  # S2: -90° to +90°
            'elbow': (0, pi),          # S3: 0° to 180°
            'wrist_pitch': (-pi, 0),    # S4: -180° to 0°
            'wrist_roll': (-pi/2, pi/2), # S5: -90° to +90°
            'gripper': (0, pi/3)        # S6: 0° to 60°
        }
        
        # Default home position (in radians)
        # Places arm in forward-facing, slightly raised position
        self.home_position = [
            pi/2,   # Base centered (90°)
            0.0,    # Shoulder horizontal (0°)
            pi/2,   # Elbow perpendicular (90°)
            -pi/2,  # Wrist pitch level (-90°)
            0.0,    # Wrist roll centered (0°)
            pi/6    # Gripper half open (30°)
        ]
        
        # DH parameters template [theta, d, a, alpha]
        # theta is variable for each joint, so it's set to 0 here
        self.dh_base_params = [
            [0, self.base_height, 0, pi/2],        # S1: Base rotation
            [pi/2, 0, self.link1_length, 0],       # S2: Shoulder (includes 90° offset)
            [0, 0, self.link2_length, 0],          # S3: Elbow
            [0, 0, self.wrist_length, -pi/2],      # S4: Wrist pitch
            [0, 0, self.wrist_offset, pi/2],       # S5: Wrist roll
            [0, 0, self.gripper_length, 0]         # S6: Gripper rotation
        ]
        
        # Calculate workspace limits
        self.max_reach = (self.link1_length + 
                         self.link2_length + 
                         self.wrist_length +
                         self.wrist_offset +
                         self.gripper_length)
        
        self.min_reach = 0.046  # Minimum reach from base axis (experimentally determined)
        
        # Dead zone parameters
        self.dead_zone_angle = pi/12  # 15° cone above base
        
        # Additional servo parameters
        self.servo_params = {
            'max_speed': pi,          # Maximum joint speed (rad/s)
            'max_acceleration': 2*pi,  # Maximum joint acceleration (rad/s²)
            'position_tolerance': 0.01 # Position tolerance (radians)
        }
    
    def get_dh_parameters(self, joint_angles: list) -> list:
        """Get Denavit-Hartenberg parameters for current joint configuration.
        
        Args:
            joint_angles: List of 6 joint angles [base, shoulder, elbow, wrist_pitch, wrist_roll, gripper]
            
        Returns:
            list: List of DH parameters [theta, d, a, alpha] for each joint
        """
        dh_params = []
        for i, base_param in enumerate(self.dh_base_params):
            # Copy base parameters and update theta
            param = base_param.copy()
            if i == 1:  # Shoulder joint
                param[0] = joint_angles[i] + pi/2  # Add 90° offset
            else:
                param[0] = joint_angles[i]
            dh_params.append(param)
        return dh_params
    
    def check_joint_limits(self, joint_angles: list) -> Tuple[bool, str]:
        """Check if joint angles are within limits.
        
        Args:
            joint_angles: List of 6 joint angles
            
        Returns:
            tuple: (is_valid, message)
        """
        if len(joint_angles) != 6:
            return False, "Expected 6 joint angles"
            
        joint_names = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
        
        for i, (name, angle) in enumerate(zip(joint_names, joint_angles)):
            min_angle, max_angle = self.joint_limits[name]
            if not (min_angle <= angle <= max_angle):
                return False, f"{name} joint angle {angle:.2f} exceeds limits [{min_angle:.2f}, {max_angle:.2f}]"
        
        return True, "Joint angles within limits"
    
    def get_servo_dimensions(self, servo_num: int) -> Dict[str, float]:
        """Get dimensions for specific servo.
        
        Args:
            servo_num: Servo number (1-6)
            
        Returns:
            dict: Dictionary containing servo dimensions
        """
        if servo_num == 6:  # Gripper servo
            return {
                'width': self.gripper_servo_width,
                'height': self.gripper_servo_height,
                'length': self.gripper_servo_length
            }
        else:  # Standard servo
            return {
                'width': self.servo_width,
                'height': self.servo_height,
                'length': self.servo_length
            }

# Create a default configuration instance
default_config = RobotConfig()
