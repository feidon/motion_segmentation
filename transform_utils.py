from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import functools

def _axis_angle_rotation(axis: str, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def euler_angles_to_matrix(euler_angles, convention: str):
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)

def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool):
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2

def matrix_to_euler_angles(matrix, convention: str):
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(matrix[..., i0, i0])
    o = (_angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
         central_angle,
         _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
        )
    return torch.stack(o, -1)


def matrix_from_angles(rot):
    """Create a rotation matrix from a triplet of rotation angles.
    Args:
    rot: a torch.Tensor of shape [..., 3], where the last dimension is the rotation
        angles, along x, y, and z.
    Returns:
    A torch.tensor of shape [..., 3, 3], where the last two dimensions are the
    rotation matrix.
    This function mimics _euler2mat from struct2depth/project.py, for backward
    compatibility, but wraps tensorflow_graphics instead of reimplementing it.
    The negation and transposition are needed to bridge the differences between
    the two.
    """
    rank = rot.dim()
    # Swap the two last dimensions
    perm = torch.cat([torch.arange(0, rank - 1, dtype=torch.long), torch.tensor([rank]), torch.tensor([rank - 1])], dim=0)
    return euler_angles_to_matrix(-rot, convention="XYZ").permute(*perm)


def angles_from_matrix(matrix):
    """Get a triplet of rotation angles from a rotation matrix.
    Args:
    matrix: A torch.tensor of shape [..., 3, 3], where the last two dimensions are
        the rotation matrix.
    Returns:
    A torch.Tensor of shape [..., 3], where the last dimension is the rotation
        angles, along x, y, and z.
    This function mimics _euler2mat from struct2depth/project.py, for backward
    compatibility, but wraps tensorflow_graphics instead of reimplementing it.
    The negation and transposition are needed to bridge the differences between
    the two.
    """
    rank = matrix.dim()
    # Swap the two last dimensions
    perm = torch.cat([torch.arange(0, rank - 2, dtype=torch.long), torch.tensor([rank - 1]), torch.tensor([rank - 2])], dim=0)
    return -matrix_to_euler_angles(matrix.permute(*perm), convention="XYZ")


def unstacked_matrix_from_angles(rx, ry, rz, name=None):
    """Create an unstacked rotation matrix from rotation angles.
    Args:
    rx: A torch.Tensor of rotation angles abound x, of any shape.
    ry: A torch.Tensor of rotation angles abound y (of the same shape as x).
    rz: A torch.Tensor of rotation angles abound z (of the same shape as x).
    name: A string, name for the op.
    Returns:
    A 3-tuple of 3-tuple of torch.Tensors of the same shape as x, representing the
    respective rotation matrix. The small 3x3 dimensions are unstacked into a
    tuple to avoid tensors with small dimensions, which bloat the TPU HBM
    memory. Unstacking is one of the recommended methods for resolving the
    problem.
    """
    angles = [-rx, -ry, -rz]
    sx, sy, sz = [torch.sin(a) for a in angles]
    cx, cy, cz = [torch.cos(a) for a in angles]
    m00 = cy * cz
    m10 = (sx * sy * cz) - (cx * sz)
    m20 = (cx * sy * cz) + (sx * sz)
    m01 = cy * sz
    m11 = (sx * sy * sz) + (cx * cz)
    m21 = (cx * sy * sz) - (sx * cz)
    m02 = -sy
    m12 = sx * cy
    m22 = cx * cy
    return ((m00, m01, m02), (m10, m11, m12), (m20, m21, m22))


def invert_rot_and_trans(rot, trans):
    """Inverts a transform comprised of a rotation and a translation.
    Args:
    rot: a torch.Tensor of shape [..., 3] representing rotatation angles.
    trans: a torch.Tensor of shape [..., 3] representing translation vectors.
    Returns:
    a tuple (inv_rot, inv_trans), representing rotation angles and translation
    vectors, such that applting rot, transm inv_rot, inv_trans, in succession
    results in identity.
    """
    inv_rot = inverse_euler(rot)  # inv_rot = -rot  for small angles
    inv_rot_mat = matrix_from_angles(inv_rot)
    inv_trans = -torch.matmul(inv_rot_mat, torch.unsqueeze(trans, -1))
    inv_trans = torch.squeeze(inv_trans, -1)
    return inv_rot, inv_trans


def inverse_euler(angles):
    """Returns the euler angles that are the inverse of the input.
    Args:
    angles: a torch.Tensor of shape [..., 3]
    Returns:
    A tensor of the same shape, representing the inverse rotation.
    """
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    sz, sy, sx = torch.unbind(-sin_angles, dim=-1)
    cz, _, cx = torch.unbind(cos_angles, dim=-1)
    y = torch.asin((cx * sy * cz) + (sx * sz))
    x = -torch.asin((sx * sy * cz) - (cx * sz)) / tf.cos(y)
    z = -torch.asin((cx * sy * sz) - (sx * cz)) / tf.cos(y)
    return torch.stack([x, y, z], dim=-1)


def combine(rot_mat1, trans_vec1, rot_mat2, trans_vec2):
    """Composes two transformations, each has a rotation and a translation.
    Args:
    rot_mat1: A torch.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec1: A torch.tensor of shape [..., 3] representing translation vectors.
    rot_mat2: A torch.tensor of shape [..., 3, 3] representing rotation matrices.
    trans_vec2: A torch.tensor of shape [..., 3] representing translation vectors.
    Returns:
    A tuple of 2 torch.Tensors, representing rotation matrices and translation
    vectors, of the same shapes as the input, representing the result of
    applying rot1, trans1, rot2, trans2, in succession.
    """
    # Building a 4D transform matrix from each rotation and translation, and
    # multiplying the two, we'd get:
    #
    # (  R2   t2) . (  R1   t1)  = (R2R1    R2t1 + t2)
    # (0 0 0  1 )   (0 0 0  1 )    (0 0 0       1    )
    #
    # Where each R is a 3x3 matrix, each t is a 3-long column vector, and 0 0 0 is
    # a row vector of 3 zeros. We see that the total rotation is R2*R1 and the t
    # total translation is R2*t1 + t2.
    r2r1 = torch.matmul(rot_mat2, rot_mat1)
    r2t1 = torch.matmul(rot_mat2, torch.unsqueeze(trans_vec1, -1))
    r2t1 = torch.squeeze(r2t1, axis=-1)
    return r2r1, r2t1 + trans_vec2