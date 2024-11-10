# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
from .kinematics import forward_kinematics, inverse_kinematics_position_only, get_joint_positions
from .config import RobotConfig, default_config

class RobotVisualizer:
    def __init__(self, config: RobotConfig = default_config):
        self.config = config
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lines = {}
        self.points = {}
        self.target_point = None
        self.ik_frames = []
        
    def _init_plot(self):
        """Initialize the 3D plot with proper scales and labels."""
        # Set axis limits based on robot dimensions
        max_reach = self.config.max_reach
        self.ax.set_xlim([-max_reach, max_reach])
        self.ax.set_ylim([-max_reach, max_reach])
        self.ax.set_zlim([0, max_reach])
        
        # Labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('SO-ARM100 IK Visualization')
        
        # Add a grid
        self.ax.grid(True)
        
    def _update_robot_plot(self, joint_positions: np.ndarray):
        """Update the robot arm visualization."""
        x = joint_positions[:, 0]
        y = joint_positions[:, 1]
        z = joint_positions[:, 2]
        
        if 'arm' not in self.lines:
            # First time plotting - create the line
            self.lines['arm'], = self.ax.plot(x, y, z, 'b-', linewidth=2, label='Robot Arm')
            self.points['joints'] = self.ax.scatter(x, y, z, c='b', marker='o')
        else:
            # Update existing line
            self.lines['arm'].set_data_3d(x, y, z)
            # For scatter plot, need to update differently
            self.points['joints'].remove()
            self.points['joints'] = self.ax.scatter(x, y, z, c='b', marker='o')
            
    def visualize_ik_solution(self, target_position: np.ndarray, 
                            initial_guess: List[float] = None,
                            max_iterations: int = 100,
                            tolerance: float = 1e-3):
        """Visualize the IK solution process with animation.
        
        Args:
            target_position: Desired end-effector position [x, y, z]
            initial_guess: Initial joint angles
            max_iterations: Maximum iterations for IK
            tolerance: Error tolerance for IK
        """
        self._init_plot()
        
        if initial_guess is None:
            initial_guess = self.config.home_position
            
        current_joints = np.array(initial_guess)
        self.ik_frames = []
        
        # Plot target position
        self.target_point = self.ax.scatter(*target_position, 
                                          c='r', marker='*', s=100,
                                          label='Target')
        
        # Store frames for animation
        for iteration in range(max_iterations):
            # Get current end-effector position
            current_position, _ = forward_kinematics(current_joints, self.config)
            joint_positions = get_joint_positions(current_joints, self.config)
            
            # Store frame
            self.ik_frames.append(joint_positions.copy())
            
            # Calculate error
            error = target_position - current_position
            error_magnitude = np.linalg.norm(error)
            
            # Check if we've reached the target
            if error_magnitude < tolerance:
                break
                
            # Update joints using IK step (simplified for visualization)
            new_joints, _ = inverse_kinematics_position_only(
                target_position,
                current_joints,
                self.config,
                max_iterations=1,
                tolerance=tolerance
            )
            current_joints = np.array(new_joints)
            
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=len(self.ik_frames),
            interval=100,
            blit=False
        )
        
        plt.legend()
        plt.show()
        
    def _animate(self, frame):
        """Animation update function."""
        self._update_robot_plot(self.ik_frames[frame])
        self.ax.set_title(f'IK Solution Process - Step {frame}')
        # Return only the line since we're handling the scatter plot separately
        return (self.lines['arm'],)
