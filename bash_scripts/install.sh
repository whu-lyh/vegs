apt-get update
apt-get install -y libgl1 libglib2.0-0 libx11-6 git

# the internet is bad, instead of installing from local disk
#pip install git+https://github.com/autonomousvision/kitti360Scripts.git
cd submodules
cd kitti360scripts
pip install -e .

cd ..

pip install submodules/diff_gaussian_rasterization
pip install submodules/simple-knn

# for image matching
#cp ../LightGlue/ckpts/superpoint_lightglue.pth /root/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth
#cp ../LightGlue/ckpts/superpoint_v1.pth /root/.cache/torch/hub/checkpoints/

# for metric evaluation
# cp ../weights/vgg16-397923af.pth /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
# cp ../weights/vgg.pth /root/.cache/torch/hub/checkpoints/vgg.pth

# for sd finetune
# cp ../weights/jx_vit_base_resnet50_384-9fd3c705.pth /root/.cache/torch/hub/checkpoints/jx_vit_base_resnet50_384-9fd3c705.pth