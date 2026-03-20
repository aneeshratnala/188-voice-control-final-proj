# test_pid.py
import numpy as np
from pid import PID  # Import your class

def test_pid():
    # Setup: Target is at 1.0, Current is at 0.0
    # Use simple gains for easy math
    kp, ki, kd = 1.0, 0.0, 0.0 
    target = np.array([1.0])
    my_pid = PID(kp, ki, kd, target)
    
    # Simulate one update step
    current_pos = np.array([0.0])
    dt = 0.1
    
    # The output should be kp * (target - current) = 1.0 * (1.0 - 0.0) = 1.0
    output = my_pid.update(current_pos, dt)
    
    print(f"Test 1 (Proportional): Expected 1.0, Got {output[0]}")
    
    # Test 2: Ensure error decreases
    if output[0] > 0:
        print("Success: Controller is pushing in the correct direction.")
    else:
        print("Fail: Controller is pushing the wrong way or not at all.")

if __name__ == "__main__":
    test_pid()