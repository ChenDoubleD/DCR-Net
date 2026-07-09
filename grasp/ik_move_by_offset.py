# ik_move_by_offset.py
# 仅提供一个函数：ik_move_by_offset_rad_simple(position, euler_rad)
# - position: [dx, dy, dz] (米)
# - euler_rad: [dphi, dtheta, dpsi] (弧度)
# 其余资源(pDll, nSocket, float_joint, POSE_c, DevMsg)从主程序(__main__)获取

import ctypes
import numpy as np
import time


def ik_move_by_offset_rad_simple(position, euler_rad):
    """
    输入:
        position  : [dx, dy, dz] (m)
        euler_rad : [dphi, dtheta, dpsi] (rad)
    返回:
        若逆解成功并执行，返回执行的关节角列表(6)；失败返回 None
    """
    # —— 从主程序(__main__)拿到已连接好的句柄与结构类型 ——
    import __main__ as _m
    try:
        pDll = _m.pDll
        nSocket = _m.nSocket
        float_joint = _m.float_joint
        POSE_c = _m.POSE_c
        DevMsg = _m.DevMsg
    except AttributeError as e:
        raise RuntimeError(
            "未从主程序找到所需对象(pDll/nSocket/float_joint/POSE_c/DevMsg)。"
            "请先在主程序完成 DLL 连接并定义这些类型后再调用本函数。"
        ) from e

    # —— 声明一次接口原型（放在这里，保证独立可用） ——
    pDll.Get_Current_Arm_State.argtypes = (
        ctypes.c_int,  # nSocket
        ctypes.c_float * 6,  # out: joints
        ctypes.POINTER(DevMsg),  # out: device pose(占位)
        ctypes.POINTER(ctypes.c_uint16),  # arm_err
        ctypes.POINTER(ctypes.c_uint16),  # sys_err
    )
    pDll.Get_Current_Arm_State.restype = ctypes.c_int

    pDll.Forward_Kinematics.argtypes = (ctypes.c_float * 6,)
    pDll.Forward_Kinematics.restype = POSE_c

    pDll.inverse_kinematics.argtypes = (ctypes.c_float * 6, ctypes.POINTER(POSE_c),
                                        ctypes.c_float * 6, ctypes.c_uint8)
    pDll.inverse_kinematics.restype = ctypes.c_int

    pDll.Movej_Cmd.argtypes = (ctypes.c_int, ctypes.c_float * 6,
                               ctypes.c_byte, ctypes.c_float, ctypes.c_bool)
    pDll.Movej_Cmd.restype = ctypes.c_int

    pDll.Set_Gripper_Pick_On.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
    pDll.Set_Gripper_Pick_On.restype = ctypes.c_int

    print("[IK] 开始逆解并移动机械臂...")

    # 1) 读取当前关节角（替代固定 joint_init）
    joint_now = float_joint()
    pose_dev = DevMsg()
    arm_err = ctypes.c_uint16(0)
    sys_err = ctypes.c_uint16(0)
    pDll.Get_Current_Arm_State(nSocket, joint_now,
                                     ctypes.byref(pose_dev),
                                     ctypes.byref(arm_err),
                                     ctypes.byref(sys_err))

    # 2) 正解得到当前末端位姿（作为叠加基准；eul 按"弧度"使用）
    pose_base = pDll.Forward_Kinematics(joint_now)

    # 3) 按你的规则构造目标
    dx, dy, dz = position
    dphi, dtheta, dpsi = euler_rad

    new_pose = POSE_c()
    new_pose.pos.x = pose_base.pos.x + dx
    new_pose.pos.y = pose_base.pos.y + dy
    new_pose.pos.z = pose_base.pos.z + dz

    new_pose.eul.phi = pose_base.eul.phi + dphi
    new_pose.eul.theta = pose_base.eul.theta + dtheta
    new_pose.eul.psi = pose_base.eul.psi + dpsi


    # 4) 逆解（种子=当前关节角）
    q_out = float_joint()
    pDll.inverse_kinematics(joint_now, ctypes.byref(new_pose), q_out, ctypes.c_uint8(1))

    # 将结果转换为列表
    target_joints = [q_out[i] for i in range(6)]

    speed = 1  # 速度百分比，可根据需要调整
    blend = 0.0  # 融合半径
    block = 1  # 阻塞模式，等待机械臂到达目标

    # 创建关节角度命令
    joint_cmd = float_joint(*[float(v) for v in target_joints])

    # 调用 Movej_Cmd
    pDll.Movej_Cmd(nSocket, joint_cmd, speed, blend, block)

    # 等待机械臂稳定
    time.sleep(1.0)

    # 6) 可选：夹爪抓取
    # pDll.Set_Gripper_Pick_On(nSocket, 500, 500, True)

    print("[IK] 机械臂移动完成")
    return target_joints