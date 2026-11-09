"""Body-frame conversion and serial-leg geometry from the loaded URDF.

No SDK dependency. All angles are URDF radians; geometry is read, never rewritten.
Kept in both standalone training packages (plane and jump).
"""
import math
import xml.etree.ElementTree as ET

import torch


def body_frame_rotation(forward_axis, device="cpu", dtype=torch.float):
    """Rows project native body vectors onto forward/left/up policy axes."""
    forward = torch.as_tensor(forward_axis, device=device, dtype=dtype)
    up = torch.tensor([0., 0., 1.], device=device, dtype=dtype)
    if forward.shape != (3,) or not torch.isfinite(forward).all():
        raise ValueError("body_forward_axis must be a finite three-vector")
    if abs(float(forward[2])) > 1e-6 or float(torch.linalg.vector_norm(forward)) < 1e-6:
        raise ValueError("Expected a nonzero horizontal forward axis with +Z up")
    forward = forward / torch.linalg.vector_norm(forward)
    left = torch.cross(up, forward, dim=0)
    return torch.stack((forward, left, up))


def _rpy_matrix(rpy, device, dtype):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # URDF fixed-axis RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll).
    return torch.tensor([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ], device=device, dtype=dtype)


class SerialLegKinematics:
    """Parallel-axis serial legs, including each joint's fixed origin rotation.

    Geometry stays in the URDF body frame. L0 and theta0 use projections onto
    policy forward/down; full 3D hip-to-wheel vectors retain lateral offsets.
    Leg array order follows the configured policy channels, RJ then LJ.
    """
    def __init__(self, urdf, policy_joint_names, device="cpu", dtype=torch.float,
                 forward_axis=(0., 1., 0.)):
        robot = ET.parse(str(urdf)).getroot()
        nodes = [j for j in robot.findall("joint") if j.get("type") != "fixed"]
        joints = {j.get("name"): j for j in nodes}
        if len(nodes) != 6 or len(joints) != 6 or set(joints) != set(policy_joint_names):
            raise ValueError("Serial-leg URDF must contain exactly the configured six DOFs")
        frame = body_frame_rotation(forward_axis, device, dtype)
        self.policy_forward, self.policy_up = frame[0], frame[2]
        policy_right = -frame[1]
        self.position_limited_names = set()
        vectors, axes, hip_positions = [], [], []
        for chain in (policy_joint_names[:3], policy_joint_names[3:]):
            chain_vectors, chain_axes = [], []
            parent = "base_link"
            orientation = torch.eye(3, device=device, dtype=dtype)
            for index, name in enumerate(chain):
                joint = joints[name]
                if joint.get("type") not in ("continuous", "revolute"):
                    raise ValueError("Expected rotary joint: " + name)
                if joint.find("parent").get("link") != parent:
                    raise ValueError("Unexpected serial joint parent: " + name)
                parent = joint.find("child").get("link")
                origin = joint.find("origin")
                xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
                axis = [float(v) for v in joint.find("axis").get("xyz").split()]
                if any(len(v) != 3 for v in (xyz, rpy, axis)) or not all(
                    math.isfinite(v) for v in xyz + rpy + axis
                ):
                    raise ValueError("Invalid joint geometry: " + name)
                position = torch.tensor(xyz, device=device, dtype=dtype)
                if index == 0:
                    hip_positions.append(position)
                else:
                    chain_vectors.append(orientation @ position)
                orientation = orientation @ _rpy_matrix(rpy, device, dtype)
                axis_body = orientation @ torch.tensor(axis, device=device, dtype=dtype)
                norm = torch.linalg.vector_norm(axis_body)
                if float(norm) < 1e-8:
                    raise ValueError("Zero joint axis: " + name)
                chain_axes.append(axis_body / norm)
                if joint.get("type") == "revolute":
                    limit = joint.find("limit")
                    if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                        raise ValueError("Missing revolute position bounds: " + name)
                    if float(limit.get("lower")) > float(limit.get("upper")):
                        raise ValueError("Reversed revolute position bounds: " + name)
                    self.position_limited_names.add(name)
            axes.append(torch.stack(chain_axes))
            vectors.append(torch.stack(chain_vectors))
        self.vectors = torch.stack(vectors)
        axes = torch.stack(axes)
        self.hip_positions = torch.stack(hip_positions)
        # The rightward axis defines a consistent orientation for each leg plane.
        normal = axes[:, 0].clone()
        alignment = torch.sum(normal * policy_right, dim=-1)
        normal *= torch.where(alignment < 0, -1., 1.).unsqueeze(-1)
        if torch.any(torch.sum(normal * policy_right, dim=-1) < 1. - 1e-6):
            raise ValueError("Leg axes do not match the configured forward/up frame")
        axis_dot = torch.sum(axes * normal[:, None, :], dim=-1)
        if torch.any(torch.abs(axis_dot) < 1. - 1e-6):
            raise ValueError("Serial-leg adapter requires parallel joint axes")
        self.joint_signs = torch.sign(axis_dot[:, :2])
        self.right_axes = normal
        up = self.policy_up - torch.sum(normal * self.policy_up, dim=-1, keepdim=True) * normal
        up = up / torch.linalg.vector_norm(up, dim=-1, keepdim=True)
        self.down_axes = -up
        self.forward_axes = torch.cross(up, normal, dim=-1)
        forward = torch.sum(self.vectors * self.forward_axes[:, None, :], dim=-1)
        down = torch.sum(self.vectors * self.down_axes[:, None, :], dim=-1)
        self.lateral_offsets = torch.sum(
            self.vectors * self.right_axes[:, None, :], dim=(-1, -2)
        )
        self.lengths = torch.sqrt(forward.square() + down.square())
        if torch.any(self.lengths < 1e-8):
            raise ValueError("Zero leg-plane link length")
        absolute_zero = torch.atan2(down, forward)
        self.zero_angles = torch.stack(
            (absolute_zero[:, 0], absolute_zero[:, 1] - absolute_zero[:, 0]), dim=-1
        )

    def joint_angles(self, hip, knee):
        theta1 = self.zero_angles[:, 0] - self.joint_signs[:, 0] * hip
        theta2 = self.zero_angles[:, 1] - self.joint_signs[:, 1] * knee
        return theta1, theta2

    def _body_position(self, theta1, theta2):
        forward = (self.lengths[:, 0] * torch.cos(theta1)
                   + self.lengths[:, 1] * torch.cos(theta1 + theta2))
        down = (self.lengths[:, 0] * torch.sin(theta1)
                + self.lengths[:, 1] * torch.sin(theta1 + theta2))
        return (forward.unsqueeze(-1) * self.forward_axes
                + down.unsqueeze(-1) * self.down_axes
                + self.lateral_offsets.unsqueeze(-1) * self.right_axes)

    def sagittal_position(self, theta1, theta2):
        body = self._body_position(theta1, theta2)
        return (torch.sum(body * self.policy_forward, dim=-1),
                -torch.sum(body * self.policy_up, dim=-1))

    def forward_kinematics(self, theta1, theta2):
        forward, down = self.sagittal_position(theta1, theta2)
        return torch.sqrt(forward.square() + down.square()), torch.atan2(-forward, down)

    def hip_to_wheel_body(self, hip, knee):
        return self._body_position(*self.joint_angles(hip, knee))
