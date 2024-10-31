from typing import NamedTuple

import numpy as np
import torch

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def trans_t_xyz(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

def combine_3dgs_rotation_translation(R_c2w, T_w2c):
    RT_w2c = np.eye(4)
    RT_w2c[:3, :3] = R_c2w.T
    RT_w2c[:3, 3] = T_w2c
    RT_c2w = np.linalg.inv(RT_w2c)
    return RT_c2w

class iComMa_input_info(NamedTuple):
    gt_pose_c2w:np.array
    start_pose_w2c:torch.tensor
    query_image:torch.tensor
    FoVx:float
    FoVy:float
    K:np.array
    image_width:int
    image_height:int

def get_pose_estimation_input(obs_view, delta):
    gt_pose_c2w = combine_3dgs_rotation_translation(obs_view.R, obs_view.T)
    start_pose_c2w = trans_t_xyz(delta[3], delta[4], delta[5]) @ \
                                rot_phi(delta[0] / 180. * np.pi) @ \
                                rot_theta(delta[1] / 180. * np.pi) @ \
                                rot_psi(delta[2] / 180. * np.pi)  @ gt_pose_c2w
    icomma_info = iComMa_input_info(gt_pose_c2w=gt_pose_c2w,
        start_pose_w2c=torch.from_numpy(np.linalg.inv(start_pose_c2w)).float(),
        query_image=obs_view.original_image[0:3, :, :],
        FoVx=obs_view.FoVx,
        FoVy=obs_view.FoVy,
        K=obs_view.K,
        image_width=obs_view.image_width,
        image_height=obs_view.image_height)
    return icomma_info

