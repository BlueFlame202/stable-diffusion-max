# Stable Diffusion with MAX

This is my project for the Modular hackathon. It sets up a wrapper around PyTorch instructions to initiate development to bring stable diffusion to MAX, with custom Mojo kernels.

The code can be run using 
```
pixi run python test.py
```

## Issues 

Sometimes, for some reason the torch which is installed doesn't come with cuda. If this is the case, please run these commands:
```
pixi run python -m pip uninstall -y torch
pixi run python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```