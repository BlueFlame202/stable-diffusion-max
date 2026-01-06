# Stable Diffusion with MAX

This is my project for the Modular hackathon. It sets up a wrapper around PyTorch instructions to initiate development to bring stable diffusion to MAX, with custom Mojo kernels.

The code can be run using 
```zsh
pixi run python test.py
```

**Project Status:** Unfinished

Over the course of the hackathon, I started exploring what it would take to bring Stable Diffusion (SDXL) inference to the MAX platform. I am quite new to hackathons, GPU programming, and everything, so the final implementation is still in early stages and not producing valid outputs yet (currently returning NaNs). The goal was to begin mapping out the architecture and figure out integration points between PyTorch-style models and MAX graphs.

So far, this has involved:
- Loading official SD v1.4 weights into a partial MAX pipeline.
- Testing an overall class wrapping the PyTorch implementations of UNet and the VAEs, so that in the future I can build out those components.
- Beginning to investigate how custom kernels might eventually be implemented in Mojo for performance-critical pieces.

**Challenges and Observations:**
- One of the big hurdles was figuring out how memory worked, as well as (slightly embarrassingly) getting confused over documentation, since I was working with the Nightly version.
- Mojo’s low-level control and MAX’s layout system are exciting, but take time to get used to—especially when working with large, complex models like SDXL.
- I didn’t yet get to implementing actual Mojo kernels or fully hooking up the graphs, but the structure is in place to support future iterations.
Despite following [the HuggingFace SDXL pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py) for inspiration (originally was trying to build SDXL), faced issues with NaN values when trying to code a simpler version.

**What’s Next:**
- Debugging the NaNs and confirming basic correctness with small test cases.
- Gradually replacing components with native MAX graphs and Mojo kernels for performance.
- Wrapping tests and benchmarking into Pixi tasks to streamline reproducibility and validation.

**Impact:**

While this submission is mainly a starting point, my hope is that it’ll be part of the start to speeding up StableDiffusion with Modular and Mojo, and making it more cross-compatible.

Thanks to the hackathon organizers and community—looking forward to continuing this work beyond the weekend!

## Issues 

Sometimes, for some reason the torch which is installed doesn't come with cuda. If this is the case, please run these commands:
```zsh
pixi run python -m pip uninstall -y torch
pixi run python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```
You might need to get `pip` first with
```zsh
pixi run python -m ensurepip --upgrade
pixi run python -m pip install --upgrade pip
```