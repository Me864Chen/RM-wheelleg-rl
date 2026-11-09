"""Resolve policy joint channels without assuming the URDF/PhysX DOF order.

Kept in both standalone training packages (plane and jump).
This maps names only; joint signs, zero offsets and physical parameters are unchanged.
"""


def resolve_joint_mapping(dof_names, policy_joint_names, joint_name_mapping):
    """Return native-order config names and policy-to-native indices.

    Native SDK tensors must stay in native order. Only policy inputs/outputs
    cross this permutation. Legacy joint names are accepted through the aliases.
    """
    native_names = list(dof_names)
    policy_names = list(policy_joint_names)
    if len(policy_names) != 6 or len(set(policy_names)) != 6:
        raise ValueError("Expected six distinct policy joint names")
    if len(native_names) != 6 or len(set(native_names)) != 6:
        raise ValueError("Expected six distinct asset DOFs, got: " + repr(native_names))
    config_names = [joint_name_mapping.get(name, name) for name in native_names]
    if len(set(config_names)) != len(config_names):
        raise ValueError("Ambiguous joint aliases: " + repr(config_names))
    missing = [name for name in policy_names if name not in config_names]
    unexpected = [name for name in config_names if name not in policy_names]
    if missing or unexpected:
        raise ValueError(
            "Joint mapping mismatch: missing={}, unexpected={}, native={}".format(
                missing, unexpected, native_names
            )
        )
    return config_names, [config_names.index(name) for name in policy_names]
