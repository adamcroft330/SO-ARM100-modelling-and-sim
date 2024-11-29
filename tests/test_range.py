# tests/test_range.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from src.config import RobotConfig
from src.kinematics import get_joint_positions, forward_kinematics
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
    
    # Create coordinate frame lines for each joint
    axis_length = config.link1_length * 0.2  # Scale axis visualization
    coord_frames = []
    for i in range(7):  # One frame for each transform (including wrist pitch linkage)
        x_axis = ax_robot.plot([], [], [], 'r-', linewidth=1, alpha=0.8)[0]  # X axis in red
        y_axis = ax_robot.plot([], [], [], 'g-', linewidth=1, alpha=0.8)[0]  # Y axis in green
        z_axis = ax_robot.plot([], [], [], 'b-', linewidth=1, alpha=0.8)[0]  # Z axis in blue
        coord_frames.append((x_axis, y_axis, z_axis))
    
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
    segments.append(ax_robot.plot([], [], [], color='#2c3e50', linewidth=6, alpha=0.8)[0])  # Base (dark blue)
    segments.append(ax_robot.plot([], [], [], color='#e74c3c', linewidth=5, alpha=0.8)[0])  # Upper arm (red)
    segments.append(ax_robot.plot([], [], [], color='#2ecc71', linewidth=4, alpha=0.8)[0])  # Forearm (green)
    segments.append(ax_robot.plot([], [], [], color='#f1c40f', linewidth=3, alpha=0.8)[0])  # Wrist pitch (yellow)
    segments.append(ax_robot.plot([], [], [], color='#9b59b6', linewidth=2, alpha=0.8)[0])  # Wrist roll (purple)
    
    # Create gripper visualization
    gripper_base = ax_robot.plot([], [], [], color='#e67e22', linewidth=2, alpha=0.8)[0]  # Gripper mount (orange)
    gripper_fixed = ax_robot.plot([], [], [], color='#1abc9c', linewidth=2, alpha=0.8)[0]  # Fixed claw (turquoise)
    gripper_moving = ax_robot.plot([], [], [], color='#1abc9c', linewidth=2, alpha=0.8)[0]  # Moving claw (turquoise)

    def update_coordinate_frame(frame_lines, transform, scale):
        """Update coordinate frame visualization based on transformation matrix."""
        origin = transform[:3, 3]
        x_axis = transform[:3, 0]
        y_axis = transform[:3, 1]
        z_axis = transform[:3, 2]
        
        # X axis (red)
        frame_lines[0].set_data_3d([origin[0], origin[0] + scale * x_axis[0]],
                                 [origin[1], origin[1] + scale * x_axis[1]],
                                 [origin[2], origin[2] + scale * x_axis[2]])
        
        # Y axis (green)
        frame_lines[1].set_data_3d([origin[0], origin[0] + scale * y_axis[0]],
                                 [origin[1], origin[1] + scale * y_axis[1]],
                                 [origin[2], origin[2] + scale * y_axis[2]])
        
        # Z axis (blue)
        frame_lines[2].set_data_3d([origin[0], origin[0] + scale * z_axis[0]],
                                 [origin[1], origin[1] + scale * z_axis[1]],
                                 [origin[2], origin[2] + scale * z_axis[2]])

    def update_robot(val=None):
        """Update robot visualization based on slider values."""
        try:
            # Get current joint angles from sliders
            joint_angles = [slider.val for slider in slider_axes]
            
            # Calculate new positions
            pos = get_joint_positions(joint_angles, config)
            
            # Get forward kinematics for orientation
            _, _, transforms = forward_kinematics(joint_angles, config, return_all_transforms=True)
            
            # Update coordinate frames for each joint
            for i, transform in enumerate(transforms[1:]):  # Skip initial identity transform
                update_coordinate_frame(coord_frames[i], transform, axis_length)
            
            # Update arm segments
            segments[0].set_data_3d([0, 0], [0, 0], [0, pos[1][2]])  # Base
            segments[1].set_data_3d([pos[1][0], pos[4][0]], [pos[1][1], pos[4][1]], [pos[1][2], pos[4][2]])  # Upper arm
            segments[2].set_data_3d([pos[4][0], pos[6][0]], [pos[4][1], pos[6][1]], [pos[4][2], pos[6][2]])  # Forearm
            segments[3].set_data_3d([pos[6][0], pos[7][0]], [pos[6][1], pos[7][1]], [pos[6][2], pos[7][2]])  # Wrist pitch
            segments[4].set_data_3d([pos[7][0], pos[8][0]], [pos[7][1], pos[8][1]], [pos[7][2], pos[8][2]])  # Wrist roll
            
            # Extract orientation vectors from roll transform for gripper
            wrist_dir = transforms[-1][:3, 0]  # X-axis after roll
            up = transforms[-1][:3, 2]         # Z-axis after roll
            
            # Update gripper
            gripper_end = pos[8] + wrist_dir * config.gripper_length
            gripper_base.set_data_3d([pos[8][0], gripper_end[0]], 
                                   [pos[8][1], gripper_end[1]], 
                                   [pos[8][2], gripper_end[2]])
            
            # Update gripper claws
            gripper_length = config.gripper_length * 0.7
            gripper_width = config.gripper_servo_width * 0.5
            curve_points = 8
            
            t = np.linspace(0, 1, curve_points)
            curve_x = gripper_length * t
            curve_y = gripper_width * np.sin(t * np.pi * 0.5)
            
            # Fixed bottom claw
            fixed_points_x = pos[8][0] + wrist_dir[0] * curve_x - up[0] * curve_y
            fixed_points_y = pos[8][1] + wrist_dir[1] * curve_x - up[1] * curve_y
            fixed_points_z = pos[8][2] + wrist_dir[2] * curve_x - up[2] * curve_y
            
            gripper_fixed.set_data_3d(fixed_points_x, fixed_points_y, fixed_points_z)
            
            # Moving top claw
            grip_angle = joint_angles[5]
            gap = gripper_width * 2 * (1 - grip_angle/config.joint_limits['gripper'][1])
            
            moving_points_x = pos[8][0] + wrist_dir[0] * curve_x + up[0] * (curve_y + gap)
            moving_points_y = pos[8][1] + wrist_dir[1] * curve_x + up[1] * (curve_y + gap)
            moving_points_z = pos[8][2] + wrist_dir[2] * curve_x + up[2] * (curve_y + gap)
            
            gripper_moving.set_data_3d(moving_points_x, moving_points_y, moving_points_z)
            
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
    
    # Add legend for coordinate axes
    ax_robot.plot([], [], [], 'r-', label='X axis', alpha=0.8)
    ax_robot.plot([], [], [], 'g-', label='Y axis', alpha=0.8)
    ax_robot.plot([], [], [], 'b-', label='Z axis', alpha=0.8)
    ax_robot.legend(loc='upper right', fontsize=8)
    
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
