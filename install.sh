conda create -n glide python=3.10 -y
conda activate glide

pip install -e .

pip install ssim
pip install lpips