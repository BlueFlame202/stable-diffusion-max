
from max.nn import (
    Module,
)

from max.graph import TensorValue, ops
from max.driver import CPU

import numpy as np

# NOTE
# the following code is largely borrowed from https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/schedulers/scheduling_pndm.py
# and subsequently modified to fit with Modular.

# TODO: write a custom op for cumprod
def cumprod(x, axis=-1):
    log_x = ops.log(x)
    cumsum_log_x = ops.cumsum(log_x, axis=axis)
    return ops.exp(cumsum_log_x)

class PNDMScheduler(Module):
    def __init__(self,
                config,
                num_train_timesteps: int = 1000,
                beta_start: float = 0.0001,
                beta_end: float = 0.02,
                beta_schedule: Literal["linear", "scaled_linear", "squaredcos_cap_v2"] = "linear",
                trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                skip_prk_steps: bool = False,
                set_alpha_to_one: bool = False,
                prediction_type: Literal["epsilon", "v_prediction"] = "epsilon",
                timestep_spacing: Literal["linspace", "leading", "trailing"] = "leading",
                steps_offset: int = 0,
    ) -> None:
        super().__init__()

        self.config = config

        if trained_betas is not None:
            self.betas = ops.constant(trained_betas, device=CPU(), dtype=DType.float32) # TODO: check that this works
        elif beta_schedule == "linear":
            self.betas = ops.range(beta_start, beta_end, (beta_end-beta_start)/num_train_timesteps, dtype=DType.float32) # TODO: check that this works
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = ops.pow(ops.range(beta_start**0.5, beta_end**0.5, (beta_end-beta_start)/num_train_timesteps, dtype=DType.float32), 2) # TODO: check that this works
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = obs.sub(1.0, self.betas)
        self.alphas_cumprod = cumprod(self.alphas, 0)

        self.final_alpha_cumprod = ops.constant(1.0, device=CPU(), dtype=DType.float32) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://huggingface.co/papers/2202.09778
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_model_output = 0
        self.counter = 0
        self.cur_sample = None
        self.ets = []

        # setable values
        self.num_inference_steps = None
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int, device=None):
        self.num_inference_steps = num_inference_steps
        if self.config.timestep_spacing == "linspace":
            self._timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps).round().astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
            self._timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            self._timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio))[::-1].astype(
                np.int64
            )
            self._timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]])[
                ::-1
            ].copy()
        else:
            prk_timesteps = np.array(self._timesteps[-self.pndm_order :]).repeat(2) + np.tile(
                np.array([0, self.config.num_train_timesteps // num_inference_steps // 2]), self.pndm_order
            )
            self.prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][
                ::-1
            ].copy()  # we copy to avoid having negative strides which are not supported by torch.from_numpy

        timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]).astype(np.int64)
        self.timesteps = ops.constant(timesteps, dtype=DType.float32, device=device)
        
        self.ets = []
        self.counter = 0
        self.cur_model_output = 0

    def scale_model_input(self, latents: TensorValue, t: TensorValue):
        pass

    def step(self, noisy_pred: TensorValue, t: TensorValue, latents: TensorValue):
        pass