from mpinets.types import PlanningProblem, ProblemSet
from geometrout.primitive import Cuboid, Cylinder, Sphere
import argparse
import pickle

def get_world_param_from_problemset(problem_chosen):
    #init nested dictionary
    world_params = {
        'world_model': {
            'coll_objs': {
                'sphere': {
                    'sphere1': {
                        'radius': 0.001,
                        'position': [10.2, -0.0, 0.4]
                    }
                },
                'cube': {
                    'cube1': {
                        'dims': [2.0, 2.0, 0.2],
                        'pose': [0.0, 0.0, -0.1, 0, 0, 0, 1.0]
                    }
                }
            }
        }
    }
    # enumerate from problem_chosen.obstacles
    sph_idx = 1
    cub_idx = 1
    cyl_idx = 1
    for idx, obs in enumerate(problem_chosen.obstacles):
        if isinstance(obs, Sphere):
            sph_key = 'sphere' + str(sph_idx)
            sph_idx += 1
            world_params['world_model']['coll_objs']['sphere'][sph_key] = {'radius': obs.radius,
                                                                           'position': obs.center}
        if isinstance(obs, Cuboid):
            cub_key = 'cube' + str(cub_idx)
            cub_idx += 1
            obs_pose = obs.center + [obs.pose.so3._quat.x,obs.pose.so3._quat.y,obs.pose.so3._quat.z,obs.pose.so3._quat.w]
            world_params['world_model']['coll_objs']['cube'][cub_key] = {'dims': obs.dims,
                                                                         'pose': obs_pose}
        if isinstance(obs, Cylinder):
            # replace cylinder with cube
            cyl_key = 'cylinder' + str(cyl_idx)
            cyl_idx += 1
            obs_pose = obs.center + [obs.pose.so3._quat.x,obs.pose.so3._quat.y,obs.pose.so3._quat.z,obs.pose.so3._quat.w]
            obs_dims = [2*obs.radius,2*obs.radius,obs.height]
            world_params['world_model']['coll_objs']['cube'][cyl_key] = {'dims': obs_dims,
                                                                         'pose': obs_pose}
    return world_params

# dont consider the cylinder due to lack of gym api
# def get_world_param_from_problemset(problem_chosen):
#     #init nested dictionary
#     world_params = {
#         'world_model': {
#             'coll_objs': {
#                 'sphere': {
#                     'sphere1': {
#                         'radius': 0.001,
#                         'position': [10.2, -0.0, 0.4]
#                     }
#                 },
#                 'cube': {
#                     'cube1': {
#                         'dims': [2.0, 2.0, 0.2],
#                         'pose': [0.0, 0.0, -0.1, 0, 0, 0, 1.0]
#                     }
#                 },
#                 'cylinder': {
#                     'cylinder1': {
#                         'radius': 0.001,
#                         'pose': [10.2, -0.0, 0.4, 0, 0, 0, 1.0],
#                         'height': 0.001
#                     }
#                 }
#             }
#         }
#     }
#     # enumerate from problem_chosen.obstacles
#     sph_idx = 1
#     cub_idx = 1
#     cyl_idx = 1
#     for idx, obs in enumerate(problem_chosen.obstacles):
#         if isinstance(obs, Sphere):
#             sph_key = 'sphere' + str(sph_idx)
#             sph_idx += 1
#             world_params['world_model']['coll_objs']['sphere'][sph_key] = {'radius': obs.radius,
#                                                                            'position': obs.center}
#         if isinstance(obs, Cuboid):
#             cub_key = 'cube' + str(cub_idx)
#             cub_idx += 1
#             obs_pose = obs.center + [obs.pose.so3._quat.x,obs.pose.so3._quat.y,obs.pose.so3._quat.z,obs.pose.so3._quat.w]
#             world_params['world_model']['coll_objs']['cube'][cub_key] = {'dims': obs.dims,
#                                                                          'pose': obs_pose}
#         if isinstance(obs, Cylinder):
#             cyl_key = 'cylinder' + str(cyl_idx)
#             cyl_idx += 1
#             obs_pose = obs.center + [obs.pose.so3._quat.x,obs.pose.so3._quat.y,obs.pose.so3._quat.z,obs.pose.so3._quat.w]
#             world_params['world_model']['coll_objs']['cylinder'][cyl_key] = {'radius': obs.radius,
#                                                                              'pose': obs_pose,
#                                                                            'height': obs.height}
#     return world_params

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "problems",
    #     type=str,
    #     default="problems/hybrid_solvable_problems.pkl",
    #     help="A pickle file of sample problems that follow the PlanningProblem format",
    # )
    # parser.add_argument(
    #     "environment_type",
    #     choices=["tabletop", "cubby", "merged_cubby", "dresser", "all"],
    #     help="The environment class",
    # )
    # parser.add_argument(
    #     "problem_type",
    #     choices=["task-oriented", "neutral_start", "neutral_goal", "all"],
    #     help="The type of planning problem",
    # )
    # args = parser.parse_args()
    # env_type = args.environment_type.replace("-", "_")
    # problem_type = args.problem_type.replace("-", "_")
    file_path = "/home/andylee/storm/mpinets/hybrid_solvable_problems.pkl"
    with open(file_path, "rb") as f:
        problems = pickle.load(f)

    # if env_type != "all":
    #     problems = {env_type: problems[env_type]}
    # if problem_type != "all":
    #     for k in problems.keys():
    #         problems[k] = {problem_type: problems[k][problem_type]}
    env_type = 'tabletop'
    problem_type = 'neutral_start'
    problem_index = 0
    problem_chosen = problems[env_type][problem_type][problem_index]
    # print(problem_chosen)
    world_params = get_world_param_from_problemset(problem_chosen)
    print(world_params)
    print("yeah")
        # if isinstance(obs, Cylinder):
        #     cub_idx = 'sphere'+str(sph_idx)
        #     cub_idx+=1
        #     world_params['world_model']['coll_objs']['cube'][cub_idx] = {'radius':obs['radius'],
        #                                                                 'position':obs['center']}