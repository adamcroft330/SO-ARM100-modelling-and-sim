# tests/test_range.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.config import RobotConfig
from src.kinematics import forward_kinematics, get_joint_positions
import matplotlib.animation as animation
from typing import List, Tuple, Optional
import time

def calculate_joint_positions(config: RobotConfig, joint_idx: int, angle: float) -> Optional[np.ndarray]:
    """Calculate joint positions for a given angle, handling invalid configurations."""
    test_config = config.home_position.copy()
    test_config[joint_idx] = angle
    try:
        return get_joint_positions(test_config, config)
    except ValueError:
        return None

def setup_3d_axis(ax: plt.Axes, title: str, max_reach: float) -> None:
    """Configure a 3D axis with consistent styling."""
    # Set title with custom styling
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold', color='#2d3436')
    
    # Set axis limits
    margin = max_reach * 0.1  # Add 10% margin
    ax.set_xlim([-max_reach-margin, max_reach+margin])
    ax.set_ylim([-max_reach-margin, max_reach+margin])
    ax.set_zlim([0, max_reach+margin])
    
    # Style axis labels
    label_style = {'fontsize': 12, 'fontweight': 'medium', 'labelpad': 15, 'color': '#2d3436'}
    ax.set_xlabel('X (m)', **label_style)
    ax.set_ylabel('Y (m)', **label_style)
    ax.set_zlabel('Z (m)', **label_style)
    
    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=10, colors='#636e72')
    
    # Set optimal view angle
    ax.view_init(elev=25, azim=45)
    
    # Set grid style
    ax.grid(True, linestyle='--', alpha=0.3, color='#b2bec3')
    
    # Set background color
    ax.set_facecolor('#f5f6fa')
    
    # Remove axis panes for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges lighter
    ax.xaxis.pane.set_edgecolor('#dfe6e9')
    ax.yaxis.pane.set_edgecolor('#dfe6e9')
    ax.zaxis.pane.set_edgecolor('#dfe6e9')


def plot_joint_range(joint_idx: int, joint_name: str, config: RobotConfig, ax: plt.Axes) -> None:
    """Plot the range of motion for a single joint."""
    print(f"Processing Joint S{joint_idx + 1}...")
    
    min_angle, max_angle = config.joint_limits[joint_name]
    test_angles = np.linspace(min_angle, max_angle, 60)
    
    positions = []
    for angle in test_angles:
        pos = calculate_joint_positions(config, joint_idx, angle)
        if pos is not None:
            positions.append(pos)
    
    if not positions:
        print(f"Warning: No valid positions found for Joint S{joint_idx + 1}")
        return
    
    positions = np.array(positions)
    
    # Create visualization elements
    line, = ax.plot([], [], [], color='#3498db', linewidth=2, alpha=0.8)
    points = ax.scatter([], [], [], color='#3498db', s=40)
    
    # Create text annotation for angle
    text_box = dict(facecolor='white', alpha=0.8, edgecolor='#dfe6e9',
                   boxstyle='round,pad=0.5')
    text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes,
                    fontsize=11, bbox=text_box, verticalalignment='top')
    
    # Plot end effector trace
    end_positions = positions[:, -1, :]
    ax.plot(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2],
            '--', color='#a4b0be', alpha=0.3, linewidth=1.5, label='End Effector Path')

    def create_roll_indicator(pos):
        if joint_idx == 4:  # S5 (wrist roll)
            s4_pos = pos[-3]  # Position of S4
            s5_pos = pos[-2]  # Position of S5
            return ax.quiver(s5_pos[0], s5_pos[1], s5_pos[2], 0.02, 0, 0, 
                           color='red', alpha=0.8)
        return None

    # Store roll indicator in a list to allow modification in nested functions
    indicators = [create_roll_indicator(positions[0])]

    def init():
        """Initialize animation with empty data."""
        line.set_data_3d([], [], [])
        points._offsets3d = ([], [], [])
        text.set_text('')
        return [line, points, text]

    def update(frame):
        """Update animation frame."""
        t = (np.sin(frame * 2 * np.pi / 200) + 1) / 2
        idx = int(t * (len(positions) - 1))
        pos = positions[idx]
        
        # Update line data for arm links
        line.set_data_3d(pos[:, 0], pos[:, 1], pos[:, 2])
        points._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # Update roll indicator for S5
        if joint_idx == 4:
            if indicators[0] is not None:
                indicators[0].remove()
            
            # Get S4-S5 linkage direction for roll axis
            s4_pos = pos[-3]
            s5_pos = pos[-2]
            roll_dir = s5_pos - s4_pos
            roll_dir = roll_dir / np.linalg.norm(roll_dir)
            
            # Calculate roll axis indicators
            angle = test_angles[idx]
            length = 0.02
            
            # Create perpendicular vectors for roll visualization
            v1 = np.array([roll_dir[1], -roll_dir[0], 0])
            if np.allclose(v1, 0):
                v1 = np.array([0, 1, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(roll_dir, v1)
            
            # Rotate the indicator around roll axis
            indicator = (v1 * np.cos(angle) + v2 * np.sin(angle)) * length
            
            # Create new indicator
            indicators[0] = ax.quiver(s5_pos[0], s5_pos[1], s5_pos[2],
                                    indicator[0], indicator[1], indicator[2],
                                    color='red', alpha=0.8)
        
        # Update angle text
        angle = test_angles[idx]
        text.set_text(f'Angle: {np.degrees(angle):.1f}°')
        
        artists = [line, points, text]
        if indicators[0] is not None:
            artists.append(indicators[0])
        return artists

    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.95, 0.95),
             framealpha=0.9, edgecolor='#dfe6e9')
    
    ax.text2D(0.02, 0.02,
             f'Range: [{np.degrees(min_angle):.1f}°, {np.degrees(max_angle):.1f}°]\n' +
             f'Neutral: {np.degrees(config.home_position[joint_idx]):.1f}°',
             transform=ax.transAxes,
             fontsize=11,
             bbox=text_box,
             verticalalignment='bottom')
    
    anim = animation.FuncAnimation(
        ax.figure, update,
        init_func=init,
        frames=400,
        interval=20,
        blit=True,
        repeat=True
    )
    
    return anim

def test_joint_ranges() -> None:
    """Visualize the range of motion for each joint of the SO-ARM100."""
    print("\nInitializing joint range visualization...")
    config = RobotConfig()
    
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('#ffffff')
    
    joint_names = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
    animations = []  # Store all animations
    
    for joint_idx, joint_name in enumerate(joint_names):
        ax = fig.add_subplot(2, 3, joint_idx + 1, projection='3d')
        setup_3d_axis(ax, f'Joint S{joint_idx + 1} Range', config.max_reach)
        anim = plot_joint_range(joint_idx, joint_name, config, ax)
        animations.append(anim)  # Keep reference to animation
    
    plt.tight_layout(pad=4.0, h_pad=5.0, w_pad=5.0)
    print("\nDisplaying joint range visualization...")
    print("Note: Window will remain open until manually closed.")
    print("Close the window to proceed to the animation.")
    plt.show()

if __name__ == "__main__":
    print("SO-ARM100 Joint Visualization Tool")
    print("==================================")
    
    try:
        test_joint_ranges()
        print("\nVisualization complete. Exiting program.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
