#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import torch
from PIL import Image
from torch import nn

from utils.general_utils import Normal2Torch
from utils.graphics_utils import (focal2fov, getProjectionMatrix,
                                  getProjectionMatrixwithPrincipalPointOffset,
                                  getWorld2View2, se3_to_SE3)


# Define a class named Camera_Pose. The code is based on the camera_transf class in iNeRF. You can refer to iNeRF at https://github.com/salykovaa/inerf.
class Camera_Pose(nn.Module):
    def __init__(self, start_pose_w2c, FoVx, FoVy, image_width, image_height,
             trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0,
             ):
        super(Camera_Pose, self).__init__()

        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.cov_offset = 0
        # infact the learnable parameters are the relative transformation and parameterized on Lie-group (SE3) manifold space
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device)) # rot
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device)) # trans
        
        self.forward(start_pose_w2c)
    
    def forward(self, start_pose_w2c):
        deltaT = se3_to_SE3(self.w, self.v)
        # pose_w2c is the updated camera pose from wrold to camera
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
        self.update()
    
    def current_campose_c2w(self):
        return self.pose_w2c.inverse().clone().cpu().detach().numpy()

    def update(self):
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class Camera_Pose2(nn.Module):
    def __init__(self, start_pose_w2c, R, T, FoVx, FoVy, K, image, image_name, image_path, image_width=None, image_height=None,
             trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
             ):
        super(Camera_Pose2, self).__init__()

        self.R = R # Transpose of R matrix of w2c
        self.T = T # t part of w2c
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K = K
        self.image_name = image_name
        self.image_path = image_path

        self.image_width = image_width
        self.image_height = image_height

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        if image is not None:
            self.image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.image.shape[2]
            self.image_height = self.image.shape[1]
            self.image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            assert image_width is not None and image_height is not None

            self.image = None
            self.image_width = image_width
            self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.cov_offset = 0
        # infact the learnable parameters are the relative transformation and parameterized on Lie-group (SE3) manifold space
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device)) # rot
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(start_pose_w2c.device)) # trans
        
        self.forward(start_pose_w2c)
    
    def forward(self, start_pose_w2c):
        deltaT = se3_to_SE3(self.w, self.v)
        # pose_w2c is the updated camera pose from wrold to camera
        # self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse() # raw iComMa, will cause the large initial error
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c)
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrixwithPrincipalPointOffset(znear=self.znear, 
                                                                             zfar=self.zfar, 
                                                                             fovX=self.FoVx, 
                                                                             fovY=self.FoVy,
                                                                             fx=self.K[0, 0],
                                                                             fy=self.K[1, 1],
                                                                             cx=self.K[0, 2],
                                                                             cy=self.K[1, 2],
                                                                             w=self.image_width,
                                                                             h=self.image_height).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def current_campose_c2w(self):
        return self.pose_w2c.inverse().clone().cpu().detach().numpy()

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, K, image, gt_alpha_mask, mask,
                 image_name, image_path, normal_path, normal, uid, frame, image_width=None, image_height=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 aug_image_size=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R # Transpose of R matrix of w2c
        self.T = T # t part of w2c
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K = K
        self.image_name = image_name
        self.image_path = image_path
        self.normal_path = normal_path
        self.normal = normal
        self.frame = frame
        self.mask = mask # lyh

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        if image is not None:
            self.image = image.clamp(0.0, 1.0).to(self.data_device)
            if mask is not None: # lyh
                self.image = self.image.permute(1, 2, 0)
                self.image[mask] = torch.FloatTensor([0, 0, 0]).to(self.data_device)
                self.image = self.image.permute(2, 0, 1)
            self.image_width = self.image.shape[2]
            self.image_height = self.image.shape[1]

            if gt_alpha_mask is not None:
                self.image *= gt_alpha_mask.to(self.data_device)
            else:
                self.image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        else:
            assert image_width is not None and image_height is not None

            self.image = None
            self.image_width = image_width
            self.image_height = image_height

        if aug_image_size is not None:
            r = aug_image_size / self.image_height
            self.K = r * self.K
            self.image_height = aug_image_size 
            self.image_width = int(r * self.image_width)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrixwithPrincipalPointOffset(znear=self.znear, 
                                                                             zfar=self.zfar, 
                                                                             fovX=self.FoVx, 
                                                                             fovY=self.FoVy,
                                                                             fx=self.K[0, 0],
                                                                             fy=self.K[1, 1],
                                                                             cx=self.K[0, 2],
                                                                             cy=self.K[1, 2],
                                                                             w=self.image_width,
                                                                             h=self.image_height).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # If preload, load them in GPU
        if self.image is not None:
            self.image = self.image.cuda()
        if self.normal is not None:
            self.normal = self.normal.cuda()

    @property
    def original_image(self):
        if self.image is not None:
            return self.image
        else:
            image = torch.from_numpy(np.array(Image.open(self.image_path))).permute(2, 0, 1) / 255.0
            return image.cuda()

    @property
    def original_normal(self):
        if self.normal is not None:
            return self.normal
        else:
            return Normal2Torch(np.load(self.normal_path), (self.image_width, self.image_height)).cuda()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def augmentCamera(viewpoint_cam, cfg_sd, eval=False):
    w2c = np.eye(4)
    w2c[:3, :3] = viewpoint_cam.R.transpose(1, 0)
    w2c[:3, 3] = viewpoint_cam.T.squeeze()
    c2w = np.linalg.inv(w2c)

    if eval:
        # Yaw (look left/right, rotate w.r.t z-axis in world coordinate)
        yaw_rad = cfg_sd.yaw_eval * np.pi/180
        R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
        
        # Pitch (look up/down)
        pitch_rad = cfg_sd.pitch_eval * np.pi/180
        R_x = np.array([[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]])

        t_z = cfg_sd.trans_z_eval
        yaw_aug_dir = None
    else:
        # Yaw (look left/right) (+ : left, - : right)
        yaw_aug_dir = np.random.choice([1, -1])
        yaw_rad = yaw_aug_dir * np.random.uniform(cfg_sd.yaw_start * np.pi/180, cfg_sd.yaw_end * np.pi/180)
        R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0], [np.sin(yaw_rad), np.cos(yaw_rad), 0], [0, 0, 1]])
        
        # Pitch (look up/down but look down only)
        pitch_rad = -1 * np.random.uniform(cfg_sd.pitch_start * np.pi/180, cfg_sd.pitch_end * np.pi/180)
        R_x = np.array([[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]])

        # Translation to y-direction (upward)
        t_z = np.random.uniform(0, cfg_sd.trans_z_range)

    # Augment
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_x) # look down in camera coordinate 
    c2w[:3, :3] = np.matmul(R_z, c2w[:3, :3]) # look left/right in world coordinate
    c2w[2, 3] = c2w[2, 3] + t_z # move upward in world coordinate z-axis

    w2c = np.linalg.inv(c2w)
    # w2c[1, 3] += t_y # move upward in camera coordinate (y is downward in OpenCV camera)

    # Return as AnnotatedCameraInstance
    viewpoint_cam_aug = Camera(colmap_id=viewpoint_cam.colmap_id, R=w2c[:3, :3].transpose(1, 0), 
                                T=w2c[:3, 3], FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy,
                                K=viewpoint_cam.K, image=viewpoint_cam.original_image, gt_alpha_mask=None,
                                image_name=viewpoint_cam.image_name, image_path=viewpoint_cam.image_path,
                                normal_path=viewpoint_cam.normal_path, normal=viewpoint_cam.normal,
                                uid=viewpoint_cam.uid, frame=viewpoint_cam.frame,
                                image_width=viewpoint_cam.image_width, image_height=viewpoint_cam.image_height, 
                                aug_image_size=cfg_sd.sd_image_size)
    return viewpoint_cam_aug, yaw_rad / np.pi * 180, pitch_rad / np.pi * 180, t_z, yaw_aug_dir

def make_camera_like_input_camera(viewpoint_cam, add_xrot_val=0, add_zrot_val=0, add_tz=0):
    '''
        add_xrot_val: rotation around x-axis in camera coordinate
        add_zrot_val: rotation around z-axis in world coordinate (why here is world?)
        add_tz: translation in z-axis in world coordinate
    '''
    w2c = np.eye(4)
    w2c[:3, :3] = viewpoint_cam.R.transpose(1, 0)
    w2c[:3, 3] = viewpoint_cam.T.squeeze()

    c2w = np.linalg.inv(w2c)

    # ext for modification - x
    phi = add_xrot_val * np.pi/180
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]) 

    # ext for modification - z
    phi = add_zrot_val * np.pi/180
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]) 
    # t_z = add_tz

    # apply 
    # c2w[0, 3] = c2w[0, 3] + t_x
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_x)

    # c2w[2, 3] = c2w[2, 3] + t_z    
    c2w[:3, :3] = np.matmul(R_z, c2w[:3, :3])
    
    c2w[2, 3] = c2w[2, 3] + add_tz

    w2c = np.linalg.inv(c2w)

    # Return as AnnotatedCameraInstance
    viewpoint_cam_new = Camera(colmap_id=viewpoint_cam.colmap_id, 
                               R=w2c[:3, :3].transpose(1, 0),  T=w2c[:3, 3], 
                               FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy, 
                               K=viewpoint_cam.K, 
                               image=viewpoint_cam.original_image,
                               normal=viewpoint_cam.normal, # FIXME CHANGE BACK TO NORMAL FOR KITTI-360 DATASET
                               gt_alpha_mask=None, mask=None,
                               image_name=viewpoint_cam.image_name,
                               image_path=viewpoint_cam.image_path,
                               normal_path=viewpoint_cam.normal_path,
                               uid=viewpoint_cam.uid,
                               frame=viewpoint_cam.frame
                               )
    return viewpoint_cam_new

def make_camera_like_input_camera_full(viewpoint_cam, 
                                    add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, 
                                    add_tx=0, add_ty=0, add_tz=0):
    '''
        add_xrot_val: rotation around x-axis in camera coordinate
        add_yrot_val: rotation around y-axis in camera coordinate
        add_zrot_val: rotation around z-axis in world coordinate
        add_tx: translation in z-axis in world coordinate
        add_ty: translation in z-axis in world coordinate
        add_tz: translation in z-axis in world coordinate
    '''
    w2c = np.eye(4)
    w2c[:3, :3] = viewpoint_cam.R.transpose(1, 0)
    w2c[:3, 3] = viewpoint_cam.T.squeeze()

    c2w = np.linalg.inv(w2c)

    # ext for modification - x
    phi = add_xrot_val * np.pi/180
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]) 

    # ext for modification - y
    theta = add_yrot_val * np.pi/180
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]) 

    # ext for modification - z
    phi = add_zrot_val * np.pi/180
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]) 

    # apply 
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_x)
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_y)
    c2w[:3, :3] = np.matmul(R_z, c2w[:3, :3])

    c2w[0, 2] = c2w[0, 2] + add_tx
    c2w[1, 2] = c2w[1, 2] + add_ty
    c2w[2, 2] = c2w[2, 2] + add_tz

    w2c = np.linalg.inv(c2w)

    # Return as AnnotatedCameraInstance
    viewpoint_cam_new = Camera(colmap_id=viewpoint_cam.colmap_id, 
                               R=w2c[:3, :3].transpose(1, 0), T=w2c[:3, 3], 
                               FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy, 
                               K=viewpoint_cam.K, 
                               image=viewpoint_cam.original_image,
                               normal=viewpoint_cam.original_normal,
                               gt_alpha_mask=None, mask=None,
                               image_name=viewpoint_cam.image_name,
                               image_path=viewpoint_cam.image_path,
                               normal_path=viewpoint_cam.normal_path,
                               uid=viewpoint_cam.uid,
                               frame=viewpoint_cam.frame
                               )
    return viewpoint_cam_new

def make_camera_like_input_camera_wo_pose(viewpoint_cam, add_xrot_val=0, add_zrot_val=0, add_tz=0):
    '''
        add_xrot_val: rotation around x-axis in camera coordinate
        add_zrot_val: rotation around z-axis in world coordinate
        add_tz: translation in z-axis in world coordinate
    '''
    w2c = np.eye(4)
    w2c[:3, :3] = viewpoint_cam.R.transpose(1, 0)
    w2c[:3, 3] = viewpoint_cam.T.squeeze()

    c2w = np.linalg.inv(w2c)

    # ext for modification - x
    phi = add_xrot_val * np.pi/180
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]) 

    # ext for modification - z
    phi = add_zrot_val * np.pi/180
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]) 
    # t_z = add_tz

    # apply 
    # c2w[0, 3] = c2w[0, 3] + t_x
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_x)

    # c2w[2, 3] = c2w[2, 3] + t_z    
    c2w[:3, :3] = np.matmul(R_z, c2w[:3, :3])
    
    c2w[2, 3] = c2w[2, 3] + add_tz

    w2c = np.linalg.inv(c2w)
    return w2c

def make_camera_like_input_camera_wo_pose_full(viewpoint_cam, 
                                               add_xrot_val=0, add_yrot_val=0, add_zrot_val=0, 
                                               add_tx=0, add_ty=0, add_tz=0):
    '''
        add_xrot_val: rotation around x-axis in camera coordinate
        add_yrot_val: rotation around y-axis in camera coordinate
        add_zrot_val: rotation around z-axis in world coordinate
        add_tx: translation in z-axis in world coordinate
        add_ty: translation in z-axis in world coordinate
        add_tz: translation in z-axis in world coordinate
    '''
    w2c = np.eye(4)
    w2c[:3, :3] = viewpoint_cam.R.transpose(1, 0)
    w2c[:3, 3] = viewpoint_cam.T.squeeze()

    c2w = np.linalg.inv(w2c)

    # ext for modification - x
    phi = add_xrot_val * np.pi/180
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]) 

    # ext for modification - y
    theta = add_yrot_val * np.pi/180
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]) 

    # ext for modification - z
    phi = add_zrot_val * np.pi/180
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]) 

    # apply 
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_x)
    c2w[:3, :3] = np.matmul(c2w[:3, :3], R_y)
    c2w[:3, :3] = np.matmul(R_z, c2w[:3, :3])

    c2w[0, 2] = c2w[0, 2] + add_tx
    c2w[1, 2] = c2w[1, 2] + add_ty
    c2w[2, 2] = c2w[2, 2] + add_tz

    w2c = np.linalg.inv(c2w)
    return w2c