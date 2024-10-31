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

import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import torch
import torchvision
from tqdm import tqdm

from arguments import (BoxModelParams, KITTI360DataParams, ModelParams,
                       PipelineParams, get_combined_args)
from gaussian_renderer import GaussianModel, render
from scene import Scene, Scene_raw
from utils.general_utils import safe_state


def save_video(images, outputfile, fps=30):
    basepath, _ = os.path.split(outputfile)
    # give writing permission to basepath
    os.system(f"chmod 777 {basepath}")
    outputVideo = cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    size = (images.shape[2], images.shape[1])
    outputVideo.open(outputfile, fourcc, fps, size, True)
    images = images.numpy()
    for image in images:
        outputVideo.write(image[..., ::-1])
    outputVideo.release() # close the writer
    return None


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    vid_images = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # vid_images.append(rendering.detach().cpu())
    # vid_images = (torch.clip(torch.stack(vid_images), 0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1)
    # save_video(vid_images, os.path.join(model_path, name, "ours_{}".format(iteration), f"renders.mp4"))


def render_sets_raw(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene_raw(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                cfg_kitti : KITTI360DataParams, cfg_box: BoxModelParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, cfg_kitti, cfg_box, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    dp = KITTI360DataParams(parser)
    bp = BoxModelParams(parser) 
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.data_type == "kitti360":
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                    dp.extract(args), bp.extract(args), args.skip_train, args.skip_test)
    else:
        render_sets_raw(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)