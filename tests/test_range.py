# tests/test_range.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from src.config import RobotConfig
from src.kinematics import get_joint_positions
from typing import List, Dict
import numpy as np

def setup_3d_axis(ax: plt.Axes, title: str, max_reach: float) -> None:
    """Configure a 3D axis with consistent styling."""
    ax.set_title(title, fontsize=12, pad=10, fontweight='bold', color='#2d3436')
    
    margin = max_reach * 0.2
    ax.set_xlim([-max_reach-margin, max_reach+margin])
    ax.set_ylim([-max_reach-margin, max_reach+margin])
    ax.set_zlim([0, max_reach+margin])
    
    label_style = {'fontsize': 10, 'fontweight': 'medium', 'labelpad': 10, 'color': '#2d3436'}
    ax.set_xlabel('X (m)', **label_style)
    ax.set_ylabel('Y (m)', **label_style)
    ax.set_zlabel('Z (m)', **label_style)
    
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#636e72')
    ax.view_init(elev=25, azim=45)
    ax.grid(True, linestyle='--', alpha=0.3, color='#b2bec3')
    ax.set_facecolor('#f5f6fa')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('#dfe6e9')
    ax.yaxis.pane.set_edgecolor('#dfe6e9')
    ax.zaxis.pane.set_edgecolor('#dfe6e9')

def create_robot_visualization():
    """Create interactive robot arm visualization with sliders."""
    config = RobotConfig()
    
    # Create figure with specific size and layout
    fig = plt.figure(figsize=(15, 8))
    
    # Create main 3D axis for robot on the right side
    ax_robot = fig.add_axes([0.35, 0.1, 0.6, 0.8], projection='3d')  # [left, bottom, width, height]
    setup_3d_axis(ax_robot, 'SO-ARM100 Joint Control', config.max_reach)
    
    # Create slider axes on the left side
    slider_axes = []
    joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Gripper']
    joint_keys = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
    
    # Position sliders vertically on the left with better spacing
    slider_width = 0.15  # Reduced width
    slider_height = 0.02
    slider_spacing = 0.12  # Increased spacing
    left_margin = 0.08
    
    # Create sliders with labels and limits
    for i in range(6):
        y_pos = 0.8 - i * slider_spacing  # Start from top
        
        # Add joint name above slider
        ax_name = fig.add_axes([left_margin, y_pos + 0.02, slider_width, 0.02])
        ax_name.axis('off')
        ax_name.text(0.5, 0.5, joint_names[i],
                    transform=ax_name.transAxes,
                    verticalalignment='center',
                    horizontalalignment='center',
                    fontsize=9,
                    fontweight='bold')
        
        # Create slider
        ax_slider = fig.add_axes([left_margin, y_pos, slider_width, slider_height])
        min_angle, max_angle = config.joint_limits[joint_keys[i]]
        init_value = config.home_position[i]
        
        # Create slider with empty label
        slider = Slider(ax_slider, '', 
                       min_angle, max_angle, 
                       valinit=init_value,
                       valstep=np.radians(1),
                       color='#3498db')
        slider.valtext.set_visible(False)  # Hide the built-in value display
        slider_axes.append(slider)
        
        # Add min limit on the left
        ax_min = fig.add_axes([left_margin - 0.06, y_pos, 0.05, slider_height])
        ax_min.axis('off')
        ax_min.text(1.0, 0.5, f"{np.degrees(min_angle):.0f}°",
                   transform=ax_min.transAxes,
                   verticalalignment='center',
                   horizontalalignment='right',
                   fontsize=8)
        
        # Add max limit on the right
        ax_max = fig.add_axes([left_margin + slider_width + 0.01, y_pos, 0.05, slider_height])
        ax_max.axis('off')
        ax_max.text(0.0, 0.5, f"{np.degrees(max_angle):.0f}°",
                   transform=ax_max.transAxes,
                   verticalalignment='center',
                   horizontalalignment='left',
                   fontsize=8)
        
        # Add current angle display below slider
        ax_angle = fig.add_axes([left_margin, y_pos - 0.02, slider_width, 0.02])
        ax_angle.axis('off')
        ax_angle.text(0.5, 0.5, f"{np.degrees(init_value):.1f}°",
                     transform=ax_angle.transAxes,
                     verticalalignment='center',
                     horizontalalignment='center',
                     fontsize=9)
        slider.angle_text = ax_angle.texts[0]  # Store reference to angle text
    
    # Create visualization elements
    segments = []
    segments.append(ax_robot.plot([], [], [], color='#3498db', linewidth=6, alpha=0.8)[0])  # Base
    segments.append(ax_robot.plot([], [], [], color='#3498db', linewidth=5, alpha=0.8)[0])  # Shoulder to elbow
    segments.append(ax_robot.plot([], [], [], color='#3498db', linewidth=4, alpha=0.8)[0])  # Elbow to wrist
    segments.append(ax_robot.plot([], [], [], color='#3498db', linewidth=3, alpha=0.8)[0])  # Wrist segments
    segments.append(ax_robot.plot([], [], [], color='#3498db', linewidth=2, alpha=0.8)[0])  # Gripper base
    
    gripper_fixed = ax_robot.plot([], [], [], color='#e17055', linewidth=2, alpha=0.8)[0]
    gripper_moving = ax_robot.plot([], [], [], color='#e17055', linewidth=2, alpha=0.8)[0]
    
    def update_robot(val=None):
        """Update robot visualization based on slider values."""
        # Get current joint angles from sliders
        joint_angles = [slider.val for slider in slider_axes]
        
        try:
            # Calculate new positions
            pos = get_joint_positions(joint_angles, config)
            
            # Update arm segments
            segments[0].set_data_3d([0, 0], [0, 0], [0, pos[1][2]])
            segments[1].set_data_3d([pos[1][0], pos[3][0]], [pos[1][1], pos[3][1]], [pos[1][2], pos[3][2]])
            segments[2].set_data_3d([pos[3][0], pos[5][0]], [pos[3][1], pos[5][1]], [pos[3][2], pos[5][2]])
            segments[3].set_data_3d([pos[5][0], pos[6][0]], [pos[5][1], pos[6][1]], [pos[5][2], pos[6][2]])
            segments[4].set_data_3d([pos[6][0], pos[7][0]], [pos[6][1], pos[7][1]], [pos[6][2], pos[7][2]])
            
            # Update gripper
            gripper_length = config.gripper_length * 0.5
            gripper_width = config.gripper_servo_width * 0.5
            
            # Calculate gripper orientation vectors
            end_dir = pos[-1] - pos[-2]
            end_dir = end_dir / np.linalg.norm(end_dir)
            
            up = np.array([0, 0, 1])
            side = np.cross(end_dir, up)
            if np.linalg.norm(side) < 1e-10:
                side = np.array([1, 0, 0])
            side = side / np.linalg.norm(side)
            
            base_pos = pos[-1]
            
            # Fixed claw
            fixed_tip = base_pos + end_dir * gripper_length - side * gripper_width
            gripper_fixed.set_data_3d([base_pos[0], fixed_tip[0]],
                                    [base_pos[1], fixed_tip[1]],
                                    [base_pos[2], fixed_tip[2]])
            
            # Moving claw
            moving_tip = base_pos + end_dir * gripper_length + side * gripper_width * np.cos(joint_angles[5])
            gripper_moving.set_data_3d([base_pos[0], moving_tip[0]],
                                     [base_pos[1], moving_tip[1]],
                                     [base_pos[2], moving_tip[2]])
            
            # Update angle displays
            for slider in slider_axes:
                slider.angle_text.set_text(f"{np.degrees(slider.val):.1f}°")
            
            fig.canvas.draw_idle()
            
        except ValueError as e:
            print(f"Invalid configuration: {str(e)}")
    
    # Connect sliders to update function
    for slider in slider_axes:
        slider.on_changed(update_robot)
    
    # Initial update
    update_robot()
    
    plt.show()

if __name__ == "__main__":
    print("SO-ARM100 Interactive Joint Control")
    print("==================================")
    print("Use sliders to control individual joints")
    print("Close the window to exit")
    
    try:
        create_robot_visualization()
        print("\nVisualization complete. Exiting program.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
