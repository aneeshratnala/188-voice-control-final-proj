import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target):
        # Initialize gains and setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.last_error = np.zeros_like(np.array(target))
        self.error_integral = np.zeros_like(np.array(target))
        # pass
        
    def reset(self, target=None):
        # Reset internal error history
        # pass
        if target is not None:
            self.target = target
        self.last_error = np.zeros_like(np.array(self.target))
        self.error_integral = np.zeros_like(np.array(self.target))
        
    def get_error(self):
        # Return magnitude of last error
        # pass
        return np.linalg.norm(self.last_error)

    def update(self, current_pos, dt):
        # Compute and return control signal
        # pass
        error = self.target - current_pos
        # P (proportional)
        p_term = self.kp * error

        # I (integral)
        self.error_integral += error * dt
        i_term = self.ki * self.error_integral

        # D (derivative)
        d_term = self.kd * (error - self.last_error) / dt

        self.last_error = error # update last error

        return p_term + i_term + d_term

        