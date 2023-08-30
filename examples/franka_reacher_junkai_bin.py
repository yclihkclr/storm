#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym 

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil

import torch
import math
torch.multiprocessing.set_start_method('spawn', force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#


import matplotlib

matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import threading
import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt

from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict

from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path

from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform, matrix_to_quaternion
from storm_kit.mpc.task.reacher_task import ReacherTask

np.set_printoptions(precision=2)

from load_problems import get_world_param_from_problemset
from mpinets.types import PlanningProblem, ProblemSet
import pickle
from transform import *

def mpc_robot_interactive(args, gym_instance):
    vis_ee_target = True
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_table.yml'
    dyn_file = 'collision_dynamic_bin.yml'

    # mpinet_problem_selection
    mpinet_problem = False
    file_path = "/home/hkclr/storm/mpinets/hybrid_solvable_problems.pkl"
    env_type = 'tabletop'
    problem_type = 'neutral_start'
    problem_index = 0
    
    gym = gym_instance.gym
    sim = gym_instance.sim
    world_yml = join_path(get_gym_configs_path(), world_file)
    dyn_yml = join_path(get_gym_configs_path(), dyn_file)   
    task_yml = join_path(mpc_configs_path(), task_file)
    robot_yml = join_path(get_gym_configs_path(), robot_file)
    

    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)
    with open(dyn_yml) as file:
        dynamic_params = yaml.load(file, Loader=yaml.FullLoader)
    with open(task_yml) as file:
        task_params = yaml.load(file, Loader=yaml.FullLoader)        
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)

    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    if (args.cuda):
        device = 'cuda'
    else:
        device = 'cpu'

    sim_params['collision_model'] = None
    # create robot simulation:
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)

    # create gym environment:
    robot_pose = sim_params['robot_pose']
    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    device = torch.device('cuda', 0)

    tensor_args = {'device': device, 'dtype': torch.float32}

    # spawn camera:
    robot_camera_pose = np.array([1.6, -1.5, 1.8, 0.707, 0.0, 0.0, 0.707])
    q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
    robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])

    robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)

    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w, w_T_r.r.x, w_T_r.r.y, w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0, 3] = w_T_r.p.x
    w_T_robot[1, 3] = w_T_r.p.y
    w_T_robot[2, 3] = w_T_r.p.z
    w_T_robot[:3, :3] = rot[0]

    # add scene from npnets problem
    if mpinet_problem:
        with open(file_path, "rb") as f:
            problems = pickle.load(f)
        problem_chosen = problems[env_type][problem_type][problem_index]
        world_params = get_world_param_from_problemset(problem_chosen)

    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)

    table_dims = np.ravel([1.5, 2.5, 0.7])
    cube_pose = np.ravel([0.35, -0.0, -0.35, 0.0, 0.0, 0.0, 1.0])

    cube_pose = np.ravel([0.9, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0])

    table_dims = np.ravel([0.35, 0.1, 0.8])

    cube_pose = np.ravel([0.35, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0])

    table_dims = np.ravel([0.3, 0.1, 0.8])

    # get camera data:
    mpc_control = ReacherTask(task_params, robot_params, world_params, tensor_args)

    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs  # rollout_fn=ArmReacher

    start_qdd = torch.zeros(n_dof, **tensor_args)

    # update goal:

    exp_params = mpc_control.exp_params

    current_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
    ee_list = []

    mpc_tensor_dtype = {'device': device, 'dtype': torch.float32}

    franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x_des_list = [franka_bl_state]

    ee_error = 10.0
    j = 0
    t_step = 0
    i = 0
    x_des = x_des_list[0]

    mpc_control.update_params(goal_state=x_des)  # first time revise goal

    # spawn object:
    x, y, z = 0.0, 0.0, 0.0
    tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002

    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(x, y, z)
    object_pose.r = gymapi.Quat(0, 0, 0, 1)

    obj_asset_file = "urdf/box/box.urdf"
    obj_asset_root = get_assets_path()

    hd_list = []
    # load ball asset
    color_b = [0.8, 0.7, 0.5]
    if('cube' in dynamic_params['world_model']['coll_objs']):
        cube = dynamic_params['world_model']['coll_objs']['cube']
        for obj in cube.keys():
            dims = cube[obj]['dims']
            pose = cube[obj]['pose']
            hd = world_instance.add_table(dims, pose, color=color_b)
            hd_1 = gym.get_actor_rigid_body_handle(env_ptr,hd,0)
            hd_list.append(hd_1)


    if (vis_ee_target):
        target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color,
                                                    name='ee_target_object')
        obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
        # obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
        gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

        obj_asset_file = "urdf/box/box.urdf"
        obj_asset_root = get_assets_path()

        ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color,
                                                name='ee_current_as_mug')
        ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
        tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

    # g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())

    # g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    # object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])

    # object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
    # object_pose = w_T_r * object_pose  # object pose in robot frame
    # if (vis_ee_target):
    #     gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    prev_acc = np.zeros(n_dof)
    ee_pose = gymapi.Transform()
    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3, 3].unsqueeze(0),
                                        rot=w_T_robot[0:3, 0:3].unsqueeze(0))

    rollout = mpc_control.controller.rollout_fn
    tensor_args = mpc_tensor_dtype
    sim_dt = mpc_control.exp_params['control_dt']

    log_traj = {'q': [], 'q_des': [], 'qdd_des': [], 'qd_des': [],
                'qddd_des': []}

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'scene_change': False}
    
    pose0_w = copy.deepcopy(world_instance.get_pose(hd_list[0]))
    pose1_w = copy.deepcopy(world_instance.get_pose(hd_list[1]))
    pose2_w = copy.deepcopy(world_instance.get_pose(hd_list[2]))
    pose3_w = copy.deepcopy(world_instance.get_pose(hd_list[3]))

    pose0_r = copy.deepcopy(w_T_r.inverse() * pose0_w)
    pose1_r = copy.deepcopy(w_T_r.inverse() * pose1_w)
    pose2_r = copy.deepcopy(w_T_r.inverse() * pose2_w)
    pose3_r = copy.deepcopy(w_T_r.inverse() * pose3_w) 

    thick = 0.05
    dynamic_params["world_model"]["coll_objs"]['cube']['obstacle1']['dims'] = [0.27, thick, 0.15]
    dynamic_params["world_model"]["coll_objs"]['cube']['obstacle2']['dims'] = [0.27, thick, 0.15]
    dynamic_params["world_model"]["coll_objs"]['cube']['obstacle3']['dims'] = [thick, 0.37, 0.15]
    dynamic_params["world_model"]["coll_objs"]['cube']['obstacle4']['dims'] = [thick, 0.37, 0.15]
    scene_change = False
    print("haaaaaaaaaaaaaaaaaaaaaaaa")
    def register_world():
        print("haaaaaaaaaaaaaaaaaaaaaaaa")
        while True:
            if nonlocal_variables['scene_change']:
                # Update obstacle position
                st = time.time()
                # pose_obs0 = nonlocal_variables['obs_pose0']
                # pose_obs1 = nonlocal_variables['obs_pose1']
                # pose_obs2 = nonlocal_variables['obs_pose2']
                # pose_obs3 = nonlocal_variables['obs_pose3']
                dynamic_params["world_model"]["coll_objs"]['cube']['obstacle1']['pose'] = [pose0_r.p.x, pose0_r.p.y, pose0_r.p.z, pose0_r.r.x, pose0_r.r.y, pose0_r.r.z, pose0_r.r.w] # w,x,y,z
                dynamic_params["world_model"]["coll_objs"]['cube']['obstacle2']['pose'] = [pose1_r.p.x, pose1_r.p.y, pose1_r.p.z, pose1_r.r.x, pose1_r.r.y, pose1_r.r.z, pose1_r.r.w]
                dynamic_params["world_model"]["coll_objs"]['cube']['obstacle3']['pose'] = [pose2_r.p.x, pose2_r.p.y, pose2_r.p.z, pose2_r.r.x, pose2_r.r.y, pose2_r.r.z, pose2_r.r.w]
                dynamic_params["world_model"]["coll_objs"]['cube']['obstacle4']['pose'] = [pose3_r.p.x, pose3_r.p.y, pose3_r.p.z, pose3_r.r.x, pose3_r.r.y, pose3_r.r.z, pose3_r.r.w]
                print()
                mpc_control.controller.rollout_fn.dynamic_collision_cost.robot_world_coll.world_coll.update_scene(
                    dynamic_params["world_model"])
                et = time.time()
                print("collision updated:", et - st)
                nonlocal_variables['scene_change'] = False
            else:
                time.sleep(0.001)

    action_thread = threading.Thread(target=register_world)
    action_thread.daemon = True
    action_thread.start()
    print("haaaaaaaaaaaaaaaaaaaaaaaa")
    q_des = None
    qd_des = None
    t_step = gym_instance.get_sim_time()

    # g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    # g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    g_pos = [0,0,0]
    g_q = [0,0,0,0]
    pre_t = gym_instance.get_sim_time()
    pre_y = 0
    while (i > -100):
        try:
            pre_t = gym_instance.get_sim_time()
            gym_instance.step()
            g_pos[0] = 0.45
            t_coef = 0.007
            g_pos[1] = 0.3*math.sin(t_coef*i)
            # g_pos[1] = 0.0
            d_gap = copy.deepcopy(g_pos[1]) - pre_y
            pre_y = copy.deepcopy(g_pos[1])
            t_gap = gym_instance.get_sim_time() - pre_t
            print("t_gap is:",t_gap)
            v_speed = d_gap/t_gap
            print("velocity of object:",v_speed)
            g_pos[2] = 0.10 
            # g_q[0] = 0.19803
            # g_q[1] = 0.6931
            # g_q[2] = 0.6931
            # g_q[3] = 0.0
            g_q[0] = 0.0
            g_q[1] = 0.7071
            g_q[2] = 0.7071
            g_q[3] = 0.0
            object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])

            object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
            object_pose = w_T_r * object_pose  # object pose in robot frame
            gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)

            pose0_w = copy.deepcopy(world_instance.get_pose(hd_list[0]))
            pose1_w = copy.deepcopy(world_instance.get_pose(hd_list[1]))
            pose2_w = copy.deepcopy(world_instance.get_pose(hd_list[2]))
            pose3_w = copy.deepcopy(world_instance.get_pose(hd_list[3]))

            pose0_r = copy.deepcopy(w_T_r.inverse() * pose0_w)
            pose1_r = copy.deepcopy(w_T_r.inverse() * pose1_w)
            pose2_r = copy.deepcopy(w_T_r.inverse() * pose2_w)
            pose3_r = copy.deepcopy(w_T_r.inverse() * pose3_w)

            # pose0_r.p.y = -0.225+0.3*math.sin(t_coef*i)
            # pose1_r.p.y = 0.225+0.3*math.sin(t_coef*i)   
            # pose2_r.p.y = 0+0.3*math.sin(t_coef*i)   
            # pose3_r.p.y = 0+0.3*math.sin(t_coef*i)

            pose0_ =np.array( [[ 0.05973631,  0.99779722, -0.02884928,  0.42036271],
                              [ 0.08055426,  0.02398802,  0.99646153,  0.45273876],
                              [ 0.99495858, -0.06184886, -0.07894386, -0.01709976],
                              [ 0.        ,  0.        ,  0.         , 1.        ]]) 

            pose1_ = np.array( [[ 0.05973631,  0.99779722, -0.02884928,  0.43103694],
                               [ 0.08055426,  0.02398802,  0.99646153,  0.084048  ],
                               [ 0.99495858, -0.06184886, -0.07894386,  0.01210947],
                               [ 0.        ,  0.        ,  0.        ,  1.        ]])

            pose2_ =np.array( [[ 0.05973631,  0.02884928,  0.99779722,  0.5627919 ],
                              [ 0.08055426, -0.99646153,  0.02398802,  0.27485393],
                              [ 0.99495858,  0.07894386, -0.06184886,  0.0289536 ],
                              [ 0.        ,  0.        ,  0.        ,  1.        ]]) 

            pose3_ =np.array( [[ 0.05973631, -0.02884928, -0.99779722,  0.29338665],
                              [ 0.08055426,  0.99646153, -0.02398802,  0.26837717],
                              [ 0.99495858, -0.07894386,  0.06184886,  0.0456528 ],
                              [ 0.        ,  0.        ,  0.        ,  1.        ]])
            pose0=pose1_
            pose1=pose0_
            pose2=pose2_
            pose3=pose3_
            
            bT_left=    np.array([ [0,0,1,0.075],
                                   [1,0,0,0    ],
                                   [0,1,0,0    ],
                                   [0,0,0,1    ]])
            
            left_T_right=np.array([[1,0,0,0     ],
                                   [0,1,0,-0.37],
                                   [0,0,1,0     ],
                                   [0,0,0,1     ]])
            
            left_T_front=np.array([[1,0,0,0.135],
                                   [0,1,0,-0.185],
                                   [0,0,1,0     ],
                                   [0,0,0,1    ]])
            
            left_T_back=np.array([ [1,0,0,-0.135],
                                   [0,1,0,-0.185],
                                   [0,0,1,0     ],
                                   [0,0,0,1     ]])

            pose0=np.dot(np.dot(pose0_,bT_left),left_T_right)
            pose1=np.dot(pose0_,bT_left)
            pose2=np.dot(np.dot(pose0_,bT_left),left_T_front)
            pose3=np.dot(np.dot(pose0_,bT_left),left_T_back)

            pose0_r.p.x=pose0[0,3]
            pose0_r.p.y=pose0[1,3]
            pose0_r.p.z=pose0[2,3]
            pose0_r.r.w=quaternion_from_matrix(pose0)[3]
            pose0_r.r.x=quaternion_from_matrix(pose0)[0]
            pose0_r.r.y=quaternion_from_matrix(pose0)[1]
            pose0_r.r.z=quaternion_from_matrix(pose0)[2]



            pose1_r.p.x=pose1[0,3]
            pose1_r.p.y=pose1[1,3]
            pose1_r.p.z=pose1[2,3]
            pose1_r.r.w=quaternion_from_matrix(pose1)[3]
            pose1_r.r.x=quaternion_from_matrix(pose1)[0]
            pose1_r.r.y=quaternion_from_matrix(pose1)[1]
            pose1_r.r.z=quaternion_from_matrix(pose1)[2]


            pose2_r.p.x=pose2[0,3]
            pose2_r.p.y=pose2[1,3]
            pose2_r.p.z=pose2[2,3]
            pose2_r.r.w=quaternion_from_matrix(pose2)[3]
            pose2_r.r.x=quaternion_from_matrix(pose2)[0]
            pose2_r.r.y=quaternion_from_matrix(pose2)[1]
            pose2_r.r.z=quaternion_from_matrix(pose2)[2]


            pose3_r.p.x=pose3[0,3]
            pose3_r.p.y=pose3[1,3]
            pose3_r.p.z=pose3[2,3]
            pose3_r.r.w=quaternion_from_matrix(pose3)[3]
            pose3_r.r.x=quaternion_from_matrix(pose3)[0]
            pose3_r.r.y=quaternion_from_matrix(pose3)[1]
            pose3_r.r.z=quaternion_from_matrix(pose3)[2]

            # revise rotation
            # pose0_r.q or 




            pose0_w = w_T_r * pose0_r
            pose1_w = w_T_r * pose1_r
            pose2_w = w_T_r * pose2_r
            pose3_w = w_T_r * pose3_r
            # rendering update
            gym.set_rigid_transform(env_ptr, hd_list[0], pose0_w)
            gym.set_rigid_transform(env_ptr, hd_list[1], pose1_w)
            gym.set_rigid_transform(env_ptr, hd_list[2], pose2_w)
            gym.set_rigid_transform(env_ptr, hd_list[3], pose3_w)

            # pose0 = copy.deepcopy(world_instance.get_pose(hd_list[0]))
            # pose1 = copy.deepcopy(world_instance.get_pose(hd_list[1]))
            # pose2 = copy.deepcopy(world_instance.get_pose(hd_list[2]))
            # pose3 = copy.deepcopy(world_instance.get_pose(hd_list[3]))   
            nonlocal_variables['scene_change'] = True
            if (vis_ee_target):
                pose = copy.deepcopy(world_instance.get_pose(obj_base_handle))
                # print("pose on robot",pose.p, pose.r)
                pose = copy.deepcopy(w_T_r.inverse() * pose)
                # print("pose on world",pose.p, pose.r)

                if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                # if (np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.1):
                    g_pos[0] = pose.p.x
                    g_pos[1] = pose.p.y + v_speed*0.2
                    g_pos[2] = pose.p.z
                    g_q[1] = pose.r.x
                    g_q[2] = pose.r.y
                    g_q[3] = pose.r.z
                    g_q[0] = pose.r.w

                    # print("object body handle:", pose.p)

                    mpc_control.update_params(goal_ee_pos=g_pos,
                                              goal_ee_quat=g_q)  # continous revise goal
            t_step += sim_dt

            #current pose
            link_pos, link_rot = mpc_control.controller.rollout_fn.dynamics_model.robot_model.get_link_pose('tcp')
            link_pos_np = link_pos.squeeze(0).cpu().numpy()
            link_quat = matrix_to_quaternion(link_rot)
            link_quat_np = link_quat.squeeze(0).cpu().numpy()
            cur_pos = link_pos_np
            # print("!!! current eef link_pos_np ",cur_pos)
            cur_quat = link_quat_np
            # print("!!! current eef link_quat_np ",cur_quat)

            current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))

            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)

            filtered_state_mpc = current_robot_state  # mpc_control.current_state
            curr_state = np.hstack(
                (filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity'])  # * 0.5
            qdd_des = copy.deepcopy(command['acceleration'])

            ee_error = mpc_control.get_current_error(filtered_state_mpc)

            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)

            # get current pose:
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])

            ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)

            if (vis_ee_target):
                gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))

            # print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt),
            #       "{:.3f}".format(mpc_control.mpc_dt))
            #

            gym_instance.clear_lines()
            top_trajs = mpc_control.top_trajs.cpu().float()  # .numpy()
            n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

            top_trajs = w_pts.cpu().numpy()
            color = np.array([0.0, 1.0, 0.0])
            for k in range(top_trajs.shape[0]):
                pts = top_trajs[k, :, :]
                color[0] = float(k) / float(top_trajs.shape[0])
                color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                gym_instance.draw_lines(pts, color=color)

            robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
            # robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            current_state = command

            i += 1

            ## record the reference trajectory
            # with open('traj_record.txt', 'ab') as f:
            #     q_des = q_des.flatten()
            #     np.savetxt(f, q_des, fmt='%.6f', newline=" ")
            #     f.write(b'\n')

        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
    mpc_control.close()
    return 1


if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()

    sim_params = load_yaml(join_path(get_gym_configs_path(), 'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)

    mpc_robot_interactive(args, gym_instance)
