PREPROCESSORS = [
    'canny',
    'depth_leres',
    'depth_leres++',
    'depth_midas',
    'depth_zoe',
    'dwpose',
    'lineart_anime',
    'lineart_coarse',
    'lineart_realistic',
    'mediapipe_face',
    'mlsd',
    'normal_bae',
    'openpose',
    'openpose_face',
    'openpose_faceonly',
    'openpose_full',
    'openpose_hand',
    'scribble_hed',
    'scribble_hedsafe',
    'scribble_pidinet',
    'scribble_pidsafe',
    'shuffle',
    'softedge_hed',
    'softedge_hedsafe',
    'softedge_pidinet',
    'softedge_pidsafe',
]

SAMPLER_NAMES = [
    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2",
    "dpm_2", "dpm_2_ancestral", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2",
    "lms", "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "gradient_estimation",
]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]

CONTROLNET_LINKS = {
    'control_v11p_sd15_inpaint.safetensors': 'https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/resolve/main/diffusion_pytorch_model.safetensors',
    'control_v11p_sd15_lineart.safetensors': 'https://huggingface.co/lllyasviel/control_v11p_sd15_lineart/resolve/main/diffusion_pytorch_model.safetensors',
    'control_v11f1p_sd15_depth.safetensors': 'https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth/resolve/main/diffusion_pytorch_model.safetensors',
    'control_v11p_sd15_openpose.safetensors': 'https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/diffusion_pytorch_model.safetensors',
}
