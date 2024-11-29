import pytest

import numpy as np
from math import pi
from src.kinematics import (transform_matrix, forward_kinematics, 
                          inverse_kinematics, calculate_jacobian)
from src.config import RobotConfig, default_config

def test_transform_matrix_input_validation():
    """Test input validation for transform_matrix function."""
    # Test NaN inputs
    with pytest.raises(ValueError, match="NaN"):
        transform_matrix(np.nan, 1.0, 1.0, 1.0)
    
    # Test infinity inputs
    with pytest.raises(ValueError, match="infinity"):
        transform_matrix(np.inf, 1.0, 1.0, 1.0)

def test_transform_matrix_identity():
    """Test transform matrix with identity case."""
    T = transform_matrix(0, 0, 0, 0)
    np.testing.assert_array_almost_equal(T, np.eye(4))

def test_forward_kinematics_input_validation():
    """Test input validation for forward kinematics."""
    # Test invalid input length
    with pytest.raises(ValueError, match="length 6"):
        forward_kinematics([1.0, 2.0, 3.0])
    
    # Test NaN inputs
    with pytest.raises(ValueError, match="NaN"):
        forward_kinematics([np.nan] * 6)
    
    # Test infinity inputs
    with pytest.raises(ValueError, match="infinity"):
        forward_kinematics([np.inf] * 6)
    
    # Test joint limits
    with pytest.raises(ValueError, match="Invalid joint angles"):
        forward_kinematics([2*pi] * 6)

def test_forward_kinematics_home_position():
    """Test forward kinematics at home position."""
    config = default_config
    position, orientation = forward_kinematics(config.home_position)
    
    # Home position should be forward and raised
    assert position[2] > 0  # Above base
    
    # Check yaw is either pi/2 or -pi/2 (equivalent orientations)
    yaw_error = min(abs(orientation[2] - pi/2), abs(orientation[2] + pi/2))
    assert yaw_error < 1e-10  # Forward facing (yaw = ±90°)
    
    assert orientation[1] < 0  # Pitched down slightly

def test_forward_kinematics_transforms():
    """Test forward kinematics with transform return."""
    config = default_config
    position, orientation, transforms = forward_kinematics(
        config.home_position, return_all_transforms=True)
    
    # Should return 7 transforms (base + 6 joints)
    assert len(transforms) == 7
    
    # First transform should be identity
    np.testing.assert_array_almost_equal(transforms[0], np.eye(4))
    
    # Last transform should match position
    np.testing.assert_array_almost_equal(transforms[-1][:3, 3], position)

def test_inverse_kinematics_convergence():
    """Test inverse kinematics convergence."""
    config = default_config
    
    # Test from home position
    initial_position, initial_orientation = forward_kinematics(config.home_position)
    
    # Try to reach same position
    joint_angles, success = inverse_kinematics(
        initial_position, initial_orientation, config.home_position)
    
    assert success, "IK should converge to home position"
    
    # Verify position
    final_position, final_orientation = forward_kinematics(joint_angles)
    np.testing.assert_array_almost_equal(final_position, initial_position, decimal=3)
    np.testing.assert_array_almost_equal(final_orientation, initial_orientation, decimal=3)

def test_jacobian_structure():
    """Test Jacobian matrix structure."""
    config = default_config
    J = calculate_jacobian(config.home_position)
    
    # Should be 6x6 matrix
    assert J.shape == (6, 6)
    
    # Should be real-valued
    assert not np.any(np.iscomplex(J))
    
    # Should be finite
    assert np.all(np.isfinite(J))

def test_jacobian_differential():
    """Test Jacobian differential relationship."""
    config = default_config
    home = config.home_position
    
    # Calculate Jacobian at home
    J = calculate_jacobian(home)
    
    # Small joint variation
    delta = 1e-6
    for i in range(6):
        # Perturb each joint slightly
        joints_plus = home.copy()
        joints_plus[i] += delta
        
        # Calculate position change
        pos1, _ = forward_kinematics(home)
        pos2, _ = forward_kinematics(joints_plus)
        
        # Compare with Jacobian prediction
        actual_delta = (pos2 - pos1)[:3]  # Only position components
        predicted_delta = J[:3, i] * delta
        
        # Should match within tolerance
        np.testing.assert_array_almost_equal(actual_delta, predicted_delta, decimal=5)

if __name__ == '__main__':
    pytest.main([__file__])
