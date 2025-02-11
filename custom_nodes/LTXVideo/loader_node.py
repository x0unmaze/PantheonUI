import json
from pathlib import Path

import pantheon
import pantheon.model_management
import pantheon.model_patcher
import folder_paths
import safetensors.torch
import torch
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from safetensors import safe_open

from .model import LTXVModel, LTXVModelConfig, LTXVTransformer3D
from .nodes_registry import pantheon_node
from .vae import LTXVVAE


@pantheon_node(name="LTXVLoader")
class LTXVLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "The name of the checkpoint (model) to load."},
                ),
                "dtype": (["bfloat16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("MODEL", "VAE")
    RETURN_NAMES = ("model", "vae")
    FUNCTION = "load"
    CATEGORY = "lightricks/LTXV"
    TITLE = "LTXV Loader"
    OUTPUT_NODE = False

    def load(self, ckpt_name, dtype):
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        load_device = pantheon.model_management.get_torch_device()
        offload_device = pantheon.model_management.unet_offload_device()

        ckpt_path = Path(folder_paths.get_full_path("checkpoints", ckpt_name))

        vae_config = None
        unet_config = None
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata is not None:
                config_metadata = metadata.get("config", None)
                if config_metadata is not None:
                    config_metadata = json.loads(config_metadata)
                    vae_config = config_metadata.get("vae", None)
                    unet_config = config_metadata.get("transformer", None)

        weights = safetensors.torch.load_file(ckpt_path, device="cpu")

        vae = self._load_vae(weights, vae_config)
        num_latent_channels = vae.first_stage_model.config.latent_channels

        model = self._load_unet(
            load_device,
            offload_device,
            weights,
            num_latent_channels,
            dtype=dtype_map[dtype],
            config=unet_config,
        )
        return (model, vae)

    def _load_vae(self, weights, config=None):
        if not config:
            config = {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3,
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "blocks": [
                    ["res_x", 4],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x", 3],
                    ["res_x", 4],
                ],
                "scaling_factor": 1.0,
                "norm_layer": "pixel_norm",
                "patch_size": 4,
                "latent_log_var": "uniform",
                "use_quant_conv": False,
                "causal_decoder": False,
            }
        vae_prefix = "vae."
        vae = LTXVVAE.from_config_and_state_dict(
            vae_class=CausalVideoAutoencoder,
            config=config,
            state_dict={
                key.removeprefix(vae_prefix): value
                for key, value in weights.items()
                if key.startswith(vae_prefix)
            },
        )
        return vae

    def _load_unet(
        self,
        load_device,
        offload_device,
        weights,
        num_latent_channels,
        dtype,
        config=None,
    ):
        if not config:
            config = {
                "_class_name": "Transformer3DModel",
                "_diffusers_version": "0.25.1",
                "_name_or_path": "PixArt-alpha/PixArt-XL-2-256x256",
                "activation_fn": "gelu-approximate",
                "attention_bias": True,
                "attention_head_dim": 64,
                "attention_type": "default",
                "caption_channels": 4096,
                "cross_attention_dim": 2048,
                "double_self_attention": False,
                "dropout": 0.0,
                "in_channels": 128,
                "norm_elementwise_affine": False,
                "norm_eps": 1e-06,
                "norm_num_groups": 32,
                "num_attention_heads": 32,
                "num_embeds_ada_norm": 1000,
                "num_layers": 28,
                "num_vector_embeds": None,
                "only_cross_attention": False,
                "out_channels": 128,
                "project_to_2d_pos": True,
                "upcast_attention": False,
                "use_linear_projection": False,
                "qk_norm": "rms_norm",
                "standardization_norm": "rms_norm",
                "positional_embedding_type": "rope",
                "positional_embedding_theta": 10000.0,
                "positional_embedding_max_pos": [20, 2048, 2048],
                "timestep_scale_multiplier": 1000,
            }

        transformer = Transformer3DModel.from_config(config)
        unet_prefix = "model.diffusion_model."
        transformer.load_state_dict(
            {
                key.removeprefix(unet_prefix): value
                for key, value in weights.items()
                if key.startswith(unet_prefix)
            }
        )
        transformer.to(dtype).to(load_device).eval()
        patchifier = SymmetricPatchifier(1)
        diffusion_model = LTXVTransformer3D(transformer, patchifier, None, None, None)
        model = LTXVModel(
            LTXVModelConfig(num_latent_channels, dtype=dtype),
            model_type=pantheon.model_base.ModelType.FLOW,
            device=pantheon.model_management.get_torch_device(),
        )
        model.diffusion_model = diffusion_model

        patcher = pantheon.model_patcher.ModelPatcher(model, load_device, offload_device)

        return patcher
