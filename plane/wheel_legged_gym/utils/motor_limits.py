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
