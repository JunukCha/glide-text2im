conda create -n glide python=3.10 -y
conda activate glide

pip install -e .

pip install lpips
pip install scikit-image
pip install torch==2.0.1 torchvision==0.15.2