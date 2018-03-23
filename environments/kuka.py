import math
import os

import numpy as np
import pybullet as p
import pybullet_data


class Kuka:
    def __init__(self, urdf_root_path=pybullet_data.getDataPath(), timestep=0.01, use_inverse_kinematics=True):
        self.urdf_root_path = urdf_root_path
        self.timestep = timestep
        self.max_velocity = .35
        self.max_force = 200.
        self.fingerA_force = 2
        self.fingerB_force = 2.5
        self.finger_tip_force = 2
        self.use_inverse_kinematics = use_inverse_kinematics
        self.use_simulation = True
        self.use_null_space = False
        self.use_orientation = True
        self.kuka_end_effector_index = 6
        self.kuka_gripper_index = 7
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.jd = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                   0.00001, 0.00001, 0.00001]
        self.kuka_uid = None
        self.reset()

    def reset(self):
        objects = p.loadSDF(os.path.join(self.urdf_root_path, "kuka_iiwa/kuka_with_gripper2.sdf"))
        self.kuka_uid = objects[0]

        p.resetBasePositionAndOrientation(self.kuka_uid, [-0.100000, 0.000000, -0.15],
                                          [0.000000, 0.000000, 0.000000, 1.000000])
        self.joint_positions = [0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
                                -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200]
        self.num_joints = p.getNumJoints(self.kuka_uid)
        for jointIndex in range(self.num_joints):
            p.resetJointState(self.kuka_uid, jointIndex, self.joint_positions[jointIndex])
            p.setJointMotorControl2(self.kuka_uid, jointIndex, p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions[jointIndex], force=self.max_force)

        self.end_effector_pos = np.array([0.537, 0.0, 0.5])
        self.end_effector_angle = 0

        self.motor_names = []
        self.motor_indices = []

        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.kuka_uid, i)
            q_index = joint_info[3]
            if q_index > -1:
                self.motor_names.append(str(joint_info[1]))
                self.motor_indices.append(i)

    def getActionDimension(self):
        if self.use_inverse_kinematics:
            return len(self.motor_indices)
        return 6  # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        state = p.getLinkState(self.kuka_uid, self.kuka_gripper_index)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def applyAction(self, motor_commands):
        if self.use_inverse_kinematics:

            dx = motor_commands[0]
            dy = motor_commands[1]
            dz = motor_commands[2]
            da = motor_commands[3]
            finger_angle = motor_commands[4]

            # Constrain effector position
            self.end_effector_pos[0] += dx
            self.end_effector_pos[0] = np.clip(self.end_effector_pos[0], 0.50, 0.65)
            self.end_effector_pos[1] += dy
            self.end_effector_pos[1] = np.clip(self.end_effector_pos[1], -0.17, 0.22)
            self.end_effector_pos[2] += dz
            self.end_effector_angle += da

            pos = self.end_effector_pos
            # Fixed orientation
            orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
            if self.use_null_space:
                if self.use_orientation:
                    joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos, orn,
                                                               self.ll, self.ul, self.jr, self.rp)
                else:
                    joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos,
                                                               lowerLimits=self.ll, upperLimits=self.ul,
                                                               jointRanges=self.jr, restPoses=self.rp)
            else:
                if self.use_orientation:
                    joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos, orn,
                                                               jointDamping=self.jd)
                else:
                    joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos)

            if self.use_simulation:
                for i in range(self.kuka_end_effector_index + 1):
                    p.setJointMotorControl2(bodyUniqueId=self.kuka_uid, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                            targetPosition=joint_poses[i], targetVelocity=0, force=self.max_force,
                                            maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)
            else:
                # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.num_joints):
                    p.resetJointState(self.kuka_uid, i, joint_poses[i])
            # fingers
            p.setJointMotorControl2(self.kuka_uid, 7, p.POSITION_CONTROL, targetPosition=self.end_effector_angle,
                                    force=self.max_force)
            p.setJointMotorControl2(self.kuka_uid, 8, p.POSITION_CONTROL, targetPosition=-finger_angle,
                                    force=self.fingerA_force)
            p.setJointMotorControl2(self.kuka_uid, 11, p.POSITION_CONTROL, targetPosition=finger_angle,
                                    force=self.fingerB_force)

            p.setJointMotorControl2(self.kuka_uid, 10, p.POSITION_CONTROL, targetPosition=0,
                                    force=self.finger_tip_force)
            p.setJointMotorControl2(self.kuka_uid, 13, p.POSITION_CONTROL, targetPosition=0,
                                    force=self.finger_tip_force)

        else:
            if self.use_simulation:
                for joint_id in range(self.kuka_end_effector_index + 1):
                    p.setJointMotorControl2(self.kuka_uid, joint_id, p.POSITION_CONTROL, targetPosition=motor_commands[joint_id],
                                            targetVelocity=0, force=self.max_force,
                                            maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)
            else:
                for i in range(self.num_joints):
                    p.resetJointState(self.kuka_uid, i, joint_poses[i])

            # fingers
            p.setJointMotorControl2(self.kuka_uid, 7, p.POSITION_CONTROL, targetPosition=motor_commands[7],
                                    force=self.max_force)
            p.setJointMotorControl2(self.kuka_uid, 8, p.POSITION_CONTROL, targetPosition=-motor_commands[8],
                                    force=self.fingerA_force)
            p.setJointMotorControl2(self.kuka_uid, 11, p.POSITION_CONTROL, targetPosition=motor_commands[8],
                                    force=self.fingerB_force)

            p.setJointMotorControl2(self.kuka_uid, 10, p.POSITION_CONTROL, targetPosition=0,
                                    force=self.finger_tip_force)
            p.setJointMotorControl2(self.kuka_uid, 13, p.POSITION_CONTROL, targetPosition=0,
                                    force=self.finger_tip_force)
