# Clone the repo
git clone https://github.com/PrimitiveX/PX-Manipulation.git

git clone https://github.com/facebookresearch/vggt.git

cd PX-Manipulation

# Create conda environment
conda create -n PX python=3.10 -y
conda activate PX

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation


# Install vggt
pip install -e path_to_vggt

# Install PX
pip install -e .
