"""Output-side actuator property overrides; no changes to source URDF files."""
import math


def apply_actuator_limits(props, names, limits):
    """Validate a complete named profile before changing any SDK properties.

    effort is Nm and velocity is rad/s, both at the URDF joint output.
    None preserves legacy asset behaviour. A profile must cover every DOF.
    """
    if limits is None:
        return props
    if len(names) != len(props) or len(set(names)) != len(names):
        raise ValueError("Actuator profile requires unique names for every DOF")
    if set(limits) != set(names):
        raise ValueError("Actuator profile must exactly match loaded joint names")
    validated = []
    for name in names:
        row = limits[name]
        if set(row) != {"effort", "velocity"}:
            raise ValueError("Expected effort and velocity for " + name)
        values = [float(row[key]) for key in ("effort", "velocity")]
        if not all(math.isfinite(v) and v > 0. for v in values):
            raise ValueError("Actuator limits must be finite and positive: " + name)
        validated.append(values)
    for i, (effort, velocity) in enumerate(validated):
        props["effort"][i] = effort
        props["velocity"][i] = velocity
    return props


def compute_pd_torques(position_target, velocity_target, position, velocity,
                       kp, kd, effort_limits, scale=1.0,
                       velocity_limits=None, position_indices=None,
                       position_bounds=None):
    """Shared batched controller for training and the standalone PD check.

    Positions/velocities/efforts are rad, rad/s and Nm at native SDK joint order.
    Optional bounds affect targets only; never overwrite physical state.
    """
    import torch

    if position_indices is not None:
        position_target = position_target.clone()
        bounds = position_bounds[position_indices]
        position_target[:, position_indices] = torch.maximum(
            torch.minimum(position_target[:, position_indices], bounds[:, 1]), bounds[:, 0]
        )
    if velocity_limits is not None:
        velocity_target = torch.clip(velocity_target, -velocity_limits, velocity_limits)
    torques = (kp * (position_target - position) + kd * (velocity_target - velocity)) * scale
    torques = torch.clip(torques, -effort_limits, effort_limits)
    if velocity_limits is not None:
        accelerating_overspeed = (velocity.abs() >= velocity_limits) & (torques * velocity > 0.)
        torques = torch.where(accelerating_overspeed, torch.zeros_like(torques), torques)
    return torques
