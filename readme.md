# SO-ARM100 Robot Arm Project 🦾

A sophisticated 6-DOF (Degrees of Freedom) robotic arm simulation and control system with full forward/inverse kinematics, visualization tools, and interactive controls.

![Robot Arm Visualization](https://via.placeholder.com/800x400?text=SO-ARM100+Robot+Arm)

## Features ✨

- **Complete 6-DOF Kinematics**
  - Forward kinematics using Denavit-Hartenberg parameters
  - Inverse kinematics with damped least squares method
  - Robust singularity handling
  - Joint limit validation

- **Interactive Visualization**
  - Real-time 3D visualization of robot pose
  - Workspace analysis and visualization
  - Joint-by-joint animation
  - Servo motor visualization with active state indication

- **React-based Control Interface**
  - Interactive joint control
  - Real-time position updates
  - Configurable view angles
  - Intuitive control panel

- **Comprehensive Testing Suite**
  - Joint range validation
  - Motion sequence testing
  - Singularity testing
  - Workspace boundary testing

## Technical Specifications 🔧

### Robot Configuration
- Base rotation: 0° to 180°
- Shoulder joint: -90° to +90°
- Elbow joint: 0° to 180°
- Wrist pitch: -180° to 0°
- Wrist roll: -90° to +90°
- Gripper: 0° to 60°

### Physical Parameters
- Maximum reach: ~12cm
- Base width: 8cm
- Link lengths: 4cm (upper arm), 4cm (forearm)
- Precision: ±0.01 radians

## Project Structure 📁

```
so-arm100/
├── src/
│   ├── config.py         # Robot configuration and parameters
│   ├── kinematics.py     # Forward/inverse kinematics implementation
│   └── visualization.py  # 3D visualization tools
├── tests/
│   ├── test_range.py     # Joint range testing
│   └── test_robot.py     # Core functionality testing
└── components/
    └── SO-ARM100 Robot Arm Visualization.tsx  # React control interface
```

## Getting Started 🚀

1. **Installation**
   ```bash
   git clone https://github.com/adamcroft330/SO-ARM100-modelling-and-sim.git
   cd SO-ARM100-modelling-and-sim
   pip install -r requirements.txt
   ```

2. **Run the Visualization**
   ```bash
   python -m src.visualization
   ```

3. **Launch the React Interface**
   ```bash
   npm install
   npm start
   ```

## Usage Examples 💡

### Basic Position Control
```python
from src.config import RobotConfig
from src.kinematics import forward_kinematics

config = RobotConfig()
joint_angles = [0, 0, 0, 0, 0, 0]  # Home position
position, orientation = forward_kinematics(joint_angles, config)
```

### Inverse Kinematics
```python
from src.kinematics import inverse_kinematics
import numpy as np

target_position = np.array([0.1, 0, 0.15])  # Target position in meters
joint_angles, success = inverse_kinematics(target_position, config=config)
```

### Visualization
```python
from src.visualization import RobotVisualizer

visualizer = RobotVisualizer()
visualizer.visualize_pose(joint_angles)
visualizer.visualize_workspace()
```

## Testing 🧪

Run the test suite to verify functionality:

```bash
python -m pytest tests/
```

## Contributing 🤝

A huge thank you to my best friend Ollie for his contributions, specifically in creating the blender model with IK and constraints in just a few hours with no robotics experience.

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 👏

- Based on standard robotics kinematics principles
- Inspired by industrial robotic arm designs
- Built with React and Three.js for visualization
- Uses numpy for mathematical computations

## Contact 📧
Email: adamcroft330@gmail.com

For questions and support, please open an issue in the GitHub repository.

---

Made by the boys.]
