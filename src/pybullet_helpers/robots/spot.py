"""Boston Dynamic Spot robot."""

from pathlib import Path

import numpy as np

from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots.single_arm import (
    FingeredSingleArmPyBulletRobot,
)
from pybullet_helpers.utils import get_assets_path
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.link import get_link_pose, get_link_state


class SpotPyBulletRobot(FingeredSingleArmPyBulletRobot):
    """Boston Dynamic Spot robot."""

    @classmethod
    def get_name(cls) -> str:
        return "spot"

    @property
    def default_urdf_path(self) -> Path:
        dir_path = get_assets_path() / "urdf"
        return dir_path / "spot" / "spot.urdf"

    @property
    def default_home_joint_positions(self) -> JointPositions:
        # Magic joint values read off spot in standard standing position with
        # arm tucked.
        return [
            0.00010371208190917969,
            -3.115184783935547,
            3.132749557495117,
            1.5715421438217163,
            -0.01901412010192871,
            -1.5716896057128906,
            -0.008634686470031738,
        ]

    @property
    def end_effector_name(self) -> str:
        return "arm_wr1"

    @property
    def tool_link_name(self) -> str:
        return "arm_link_wr1"

    @property
    def finger_joint_names(self) -> list[str]:
        return ["arm_f1x"]

    @property
    def open_fingers_state(self) -> float:
        return -np.pi / 2

    @property
    def closed_fingers_state(self) -> float:
        return 0.0

    def finger_state_to_joints(self, state: float) -> list[float]:
        return [state]

    def joints_to_finger_state(self, joint_positions: list[float]) -> float:
        assert len(joint_positions) == 1
        finger_state = joint_positions[0]
        return finger_state
    
    @property
    def default_inverse_kinematics_method(self) -> str:
        return "custom"

    def custom_inverse_kinematics(
        self,
        end_effector_pose: Pose,
        validate: bool = True,
        best_effort: bool = False,
        validation_atol: float = 1e-3,
    ) -> JointPositions:
        """Analytic IK for 6-DOF spot arm.
        
        I believe the original author of this is Tomas Lozano-Perez.
        """
        # The target end_effector_pose is in the world frame. Convert it into the
        # robot shoulder frame.
        world_to_base = self.get_base_pose()
        world_to_shoulder = multiply_poses(world_to_base, Pose((0.292, 0., 0.873)))
        shoulder_to_target = multiply_poses(world_to_shoulder.invert(), end_effector_pose)
        solns = analytic_spot_ik_6(shoulder_to_target.to_matrix(), self.joint_lower_limits[:6], self.joint_upper_limits[:6])
        # Arbitrarily select the first one. Later, might want to use distances.
        assert solns
        return list(solns[0]) + [self.get_finger_state()]



def analytic_spot_ik_6(wrist_pose_rel_shoulder, min_limits, max_limits):
    px, py, pz = wrist_pose_rel_shoulder[:3, 3]   # wrist position
    l2 = 0.3385
    l3 = np.sqrt(0.40330**2 + 0.0750**2)
    q3_off = np.arctan2(0.0750, 0.40330)
    # Solve the first three joints based on position
    q123_sols = []
    xl = np.sqrt(px**2 + py**2)
    q23 = IK2R(l2, l3, xl, -pz)
    if q23 is None:
        return []
    q1 = np.arctan2(py, px)
    for (q2, q3) in q23:
        q123_sols.append((q1, q2, q3 + q3_off))
    q23 = IK2R(l2, l3, -xl, -pz)
    assert q23 is not None
    q1 = q1 + np.pi
    for (q2, q3) in q23:
        q123_sols.append((q1, q2, q3 + q3_off))

    # Solve for the wrist angles
    qfull_sols = []
    for (q1, q2, q3) in q123_sols:
        r3_inv = inverse_matrix(
                  concatenate_matrices(rotation_matrix(q1, (0, 0, 1)),
                                       rotation_matrix(q2+q3, (0, 1, 0))))
        W = concatenate_matrices(r3_inv, wrist_pose_rel_shoulder)
        q5 = np.arccos(W[0, 0])
        for q in (q5, -q5):
            s5 = np.sin(q)
            q4 = np.arctan2(W[1, 0]/s5, -W[2, 0]/s5)
            q6 = np.arctan2(W[0, 1]/s5, W[0, 2]/s5)
            angles = (q1, q2, q3, q4, q, q6)
            if all_between(min_limits, angles, max_limits):
                qfull_sols.append(angles)
    return qfull_sols


def IK2R(L1, L2, x, y):
    xy_sq = x**2 + y**2
    c2 = (xy_sq - L1**2 - L2**2)/(2*L1*L2)
    if abs(c2) > 1:
        return None
    elif c2 == 1.0:
        return [(np.arctan2(y, x), 0)]
    elif c2 == -1.0 and xy_sq != 0.:
        return [(np.arctan2(y, x), np.pi)]
    elif c2 == -1.0 and xy_sq == 0.:
        return [(q1, np.pi) for q1 in [0, 2*np.pi]]
    else:
        q2_1 = np.arccos(c2)
        q2_2 = -q2_1
        theta = np.arctan2(y, x)
        q1q2 = [(theta - np.arctan2(L2*np.sin(q2_i), L1+L2*np.cos(q2_i)), q2_i) \
                for q2_i in (q2_1, q2_2)]
        for q1, q2 in q1q2:
            xk = (L1*np.cos(q1) + L2*np.cos(q1 + q2))
            yk = (L1*np.sin(q1) + L2*np.sin(q1 + q2))
            assert abs(x - xk) < 0.0001
            assert abs(y - yk) < 0.0001
        return q1q2


def angle_diff(x, y):
    twoPi = 2*np.pi
    z = (x - y) % twoPi
    return z - twoPi if z > np.pi else z

        
def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()


def inverse_matrix(matrix):
    """Return inverse of square transformation matrix.
    >>> M0 = random_rotation_matrix()
    >>> M1 = inverse_matrix(M0.T)
    >>> np.allclose(M1, np.linalg.inv(M0.T))
    True
    >>> for size in range(1, 7):
    ...     M0 = np.random.rand(size, size)
    ...     M1 = inverse_matrix(M0)
    ...     if not np.allclose(M1, np.linalg.inv(M0)): print size
    """
    return np.linalg.inv(matrix)



def concatenate_matrices(*matrices):
    """Return concatenation of series of transformation matrices.
    >>> M = np.random.rand(16).reshape((4, 4)) - 0.5
    >>> np.allclose(M, concatenate_matrices(M))
    True
    >>> np.allclose(np.dot(M, M.T), concatenate_matrices(M, M.T))
    True
    """
    M = np.identity(4)
    for i in matrices:
        M = np.dot(M, i)
    return M


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2., np.trace(rotation_matrix(math.pi/2,
    ...                                                direc, point)))
    True
    """
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.
    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3), dtype=np.float64)
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1.0]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
