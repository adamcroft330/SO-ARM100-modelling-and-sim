# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict
from .kinematics import forward_kinematics, inverse_kinematics, get_joint_positions, transform_matrix
from .config import RobotConfig, default_config

class RobotVisualizer:
    def __init__(self, config: RobotConfig = default_config):
        """Initialize the robot visualizer with given configuration."""
        self.config = config
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lines = {}
        self.patches = []
        self.labels = {}
        self.target_point = None
        
    def _init_plot(self):
        """Initialize the 3D plot with proper scales and labels."""
        # Clear existing plot
        self.ax.cla()
        
        # Set axis limits based on robot dimensions
        max_reach = self.config.max_reach
        self.ax.set_xlim([-max_reach, max_reach])
        self.ax.set_ylim([-max_reach, max_reach])
        self.ax.set_zlim([0, max_reach])
        
        # Labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('SO-ARM100 Robot Arm')
        
        # Add a grid
        self.ax.grid(True)
        
        # Set viewing angle for better visualization
        self.ax.view_init(elev=30, azim=45)
        
    def _create_servo_vertices(self, servo_num: int) -> np.ndarray:
        """Create vertices for a servo motor based on its dimensions."""
        dims = self.config.get_servo_dimensions(servo_num)
        w, h, l = dims['width'], dims['height'], dims['length']
        
        # Create box vertices centered at origin
        vertices = np.array([
            [-w/2, -h/2, -l/2],  # 0
            [w/2, -h/2, -l/2],   # 1
            [w/2, h/2, -l/2],    # 2
            [-w/2, h/2, -l/2],   # 3
            [-w/2, -h/2, l/2],   # 4
            [w/2, -h/2, l/2],    # 5
            [w/2, h/2, l/2],     # 6
            [-w/2, h/2, l/2]     # 7
        ])
        
        return vertices
        
    def _draw_servo(self, position: np.ndarray, orientation: np.ndarray, 
                   servo_num: int, is_active: bool = False):
        """Draw a 3D representation of a servo motor."""
        vertices = self._create_servo_vertices(servo_num)
        
        # Transform vertices to world coordinates
        transformed_vertices = (orientation @ vertices.T).T + position
        
        # Define faces (each face is a list of vertex indices)
        faces = [
            [0, 1, 2, 3],  # Front
            [4, 5, 6, 7],  # Back
            [0, 1, 5, 4],  # Bottom
            [2, 3, 7, 6],  # Top
            [0, 3, 7, 4],  # Left
            [1, 2, 6, 5]   # Right
        ]
        
        # Colors for active and inactive servos
        colors = ['#ff4444', '#cc2222', '#dd3333', '#dd3333', '#ee3333', '#ee3333'] if is_active else \
                ['#666666', '#444444', '#555555', '#555555', '#5a5a5a', '#5a5a5a']
        
        # Draw each face
        for face, color in zip(faces, colors):
            xs = [transformed_vertices[i][0] for i in face]
            ys = [transformed_vertices[i][1] for i in face]
            zs = [transformed_vertices[i][2] for i in face]
            
            # Add first point again to close the polygon
            xs.append(xs[0])
            ys.append(ys[0])
            zs.append(zs[0])
            
            # Plot face
            self.ax.plot_surface(
                np.array([xs[:-1]]),
                np.array([ys[:-1]]),
                np.array([zs[:-1]]),
                color=color,
                alpha=0.8
            )
            
        # Add servo label
        self.ax.text(position[0], position[1], position[2], 
                    f'S{servo_num}',
                    color='white',
                    fontsize=8,
                    horizontalalignment='center',
                    verticalalignment='center')
        
    def _calculate_servo_orientations(self, joint_angles: List[float]) -> List[np.ndarray]:
        """Calculate orientation matrices for each servo."""
        orientations = []
        T = np.eye(4)
        
        # Get DH parameters
        dh_params = self.config.get_dh_parameters(joint_angles)
        
        # Special orientation for base servo
        base_orientation = np.eye(3)
        orientations.append(base_orientation)
        
        # Calculate orientations for other servos
        for i, params in enumerate(dh_params[:-1]):  # Exclude gripper
            T = T @ transform_matrix(*params)
            orientation = T[:3, :3]
            
            # Adjust orientation based on servo mounting
            if i == 1:  # Shoulder
                orientation = orientation @ np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]
                ])
            elif i == 5:  # Gripper servo
                orientation = orientation @ np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]
                ])
            
            orientations.append(orientation)
            
        return orientations
    
    def visualize_pose(self, joint_angles: List[float], active_joint: int = None):
        """Visualize a single robot pose."""
        self._init_plot()
        
        # Get joint positions
        positions = get_joint_positions(joint_angles, self.config)
        
        # Draw base platform
        base_radius = self.config.base_width / 2
        theta = np.linspace(0, 2*np.pi, 32)
        x = base_radius * np.cos(theta)
        y = base_radius * np.sin(theta)
        self.ax.plot_surface(
            np.array([x]),
            np.array([y]),
            np.zeros((1, 32)),
            color='gray',
            alpha=0.5
        )
        
        # Calculate servo orientations
        orientations = self._calculate_servo_orientations(joint_angles)
        
        # Draw links
        for i in range(len(positions)-1):
            start = positions[i]
            end = positions[i+1]
            self.ax.plot([start[0], end[0]], 
                        [start[1], end[1]], 
                        [start[2], end[2]], 
                        color='#3b82f6' if i+1 == active_joint else '#60a5fa',
                        linewidth=2)
        
        # Draw servos
        for i, (pos, orient) in enumerate(zip(positions[:-1], orientations)):
            self._draw_servo(pos, orient, i+1, is_active=(i+1 == active_joint))
        
        plt.show()
    
    def animate_movement(self, start_angles: List[float], end_angles: List[float], 
                        duration: float = 2.0, fps: int = 30):
        """Animate movement between two configurations."""
        frames = int(duration * fps)
        
        def update(frame):
            # Clear previous frame
            self.ax.cla()
            self._init_plot()
            
            # Interpolate joint angles
            t = frame / (frames - 1)
            current_angles = [
                start + t * (end - start)
                for start, end in zip(start_angles, end_angles)
            ]
            
            # Visualize current pose
            self.visualize_pose(current_angles)
            
            return self.patches
        
        ani = FuncAnimation(
            self.fig, update,
            frames=frames,
            interval=1000/fps,
            blit=True
        )
        
        plt.show()
        
    def visualize_workspace(self, samples: int = 1000):
        """Visualize the robot's workspace by sampling random configurations."""
        self._init_plot()
        
        points = []
        for _ in range(samples):
            # Generate random joint angles within limits
            angles = []
            for joint in ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']:
                min_angle, max_angle = self.config.joint_limits[joint]
                angles.append(np.random.uniform(min_angle, max_angle))
            
            # Calculate end-effector position
            try:
                position, _ = forward_kinematics(angles, self.config)
                points.append(position)
            except ValueError:
                continue
        
        # Convert points to numpy array
        points = np.array(points)
        
        # Plot workspace points
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c='b', alpha=0.1, s=1)
        
        # Add workspace boundary sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        r = self.config.max_reach
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_surface(x, y, z, color='b', alpha=0.1)
        
        plt.show()

    def visualize_ik_solution(self, target_position: np.ndarray,
                            target_orientation: np.ndarray = None,
                            initial_guess: List[float] = None):
        """Visualize the IK solution process."""
        self._init_plot()
        
        # Plot target position
        self.ax.scatter(*target_position, c='r', marker='*', s=100, label='Target')
        
        # Calculate IK solution
        joint_angles, success = inverse_kinematics(
            target_position,
            target_orientation,
            initial_guess,
            self.config
        )
        
        if success:
            # Visualize final pose
            self.visualize_pose(joint_angles)
            plt.title('IK Solution Found')
        else:
            plt.title('IK Solution Failed')
            
        plt.legend()
        plt.show()
