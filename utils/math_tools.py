import numpy as np

def collective_thrust_vectorized(C_T: float, rpm: np.ndarray) -> np.ndarray:
    """
    Vectorized collective thrust for all timesteps.

    Parameters
    ----------
    C_T : float         thrust coefficient [N/rpm²]
    rpm : (N, 6)        per-motor RPM

    Returns
    -------
    f_col : (N,)        collective thrust [N]
    """
    return C_T * np.sum(rpm.astype(np.float64) ** 2, axis=1)

def quaternion_to_euler_vectorized(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Quaternion → (roll, pitch) for all timesteps.  ZYX convention.

    Parameters
    ----------
    quaternion : (N, 4)  [qw, qx, qy, qz]

    Returns
    -------
    roll  : (N,)  [rad]
    pitch : (N,)  [rad]
    """
    qw = quaternion[:, 0]
    qx = quaternion[:, 1]
    qy = quaternion[:, 2]
    qz = quaternion[:, 3]

    # roll  (φ) = atan2(2(qw*qx + qy*qz), 1 - 2(qx² + qy²))
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (θ) = asin(2(qw*qy - qz*qx))  clamped to [-1, 1]
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    return roll, pitch


def rpm_to_moments_vectorized(
    C_T: float,
    rpm: np.ndarray,
    arm_length: float = 0.265,
    motor_angles_deg: np.ndarray = None,
) -> np.ndarray:
    """
    Compute roll/pitch moments from individual motor RPMs.

    S550 hex motor layout (from front, CW):
        Motor 1:  30°  (front)
        Motor 2:  90°  (front-right)
        Motor 3: 150°  (back-right)
        Motor 4: 210°  (back)
        Motor 5: 270°  (back-left)
        Motor 6: 330°  (front-left)

    τ_x (roll)  =  L · Σ f_i · sin(θ_i)
    τ_y (pitch) = -L · Σ f_i · cos(θ_i)

    Parameters
    ----------
    C_T : float             thrust coefficient [N/rpm²]
    rpm : (N, 6)            per-motor RPM
    arm_length : float      arm length [m]
    motor_angles_deg : (6,) motor angles [deg], default hex layout

    Returns
    -------
    moments : (N, 2)  [τ_x, τ_y]  roll and pitch moments [N·m]
    """
    if motor_angles_deg is None:
        motor_angles_deg = np.array([30, 90, 150, 210, 270, 330])

    angles = np.radians(motor_angles_deg)  # (6,)
    L = arm_length

    # Per-motor thrust: (N, 6)
    f_motors = C_T * rpm.astype(np.float64) ** 2

    # Moment arms
    cos_a = np.cos(angles)   # (6,)
    sin_a = np.sin(angles)   # (6,)

    # τ_x = L · Σ f_i · sin(θ_i)
    tau_x = L * (f_motors @ sin_a)    # (N,)
    # τ_y = -L · Σ f_i · cos(θ_i)
    tau_y = -L * (f_motors @ cos_a)   # (N,)

    return np.column_stack([tau_x, tau_y])