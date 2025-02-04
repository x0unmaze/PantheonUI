import folder_paths
import pantheon.utils
import pantheon.model_detection
import pantheon.model_management
import pantheon.lora
from pantheon.model_patcher import ModelPatcher

from .utils import TimestepKeyframeGroup
from .control import ControlNetAdvanced, load_controlnet




def convert_cn_lora_from_diffusers(cn_model: ModelPatcher, lora_path: str):
    lora_data = pantheon.utils.load_torch_file(lora_path, safe_load=True)
    unet_dtype = pantheon.model_management.unet_dtype()
    for key, value in lora_data.items():
        lora_data[key] = value.to(unet_dtype)
    diffusers_keys = pantheon.utils.unet_to_diffusers(cn_model.model.state_dict())

    #lora_data = pantheon.model_detection.unet_config_from_diffusers_unet(lora_data, dtype=unet_dtype)



    #key_map = pantheon.lora.model_lora_keys_unet(cn_model.model, key_map)
    lora_data = pantheon.lora.load_lora(lora_data, to_load=diffusers_keys)

    # TODO: detect if diffusers for sure? not sure if needed at this time, since cn loras are
    # only used currently for LOOSEControl, and those are all in diffusers format
    #unet_dtype = pantheon.model_management.unet_dtype()
    #lora_data = pantheon.model_detection.unet_config_from_diffusers_unet(lora_data, unet_dtype)
    return lora_data


class ControlNetLoaderWithLoraAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
                "cn_lora_name": (folder_paths.get_filename_list("controlnet"), ),
                "cn_lora_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "timestep_keyframe": ("TIMESTEP_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/LOOSEControl"

    def load_controlnet(self, control_net_name, cn_lora_name, cn_lora_strength: float,
                        timestep_keyframe: TimestepKeyframeGroup=None
                        ):
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet: ControlNetAdvanced = load_controlnet(controlnet_path, timestep_keyframe)
        if not isinstance(controlnet, ControlNetAdvanced):
            raise ValueError("Type {} is not compatible with CN LoRA features at this time.")
        # now, try to load CN LoRA
        lora_path = folder_paths.get_full_path("controlnet", cn_lora_name)
        lora_data = convert_cn_lora_from_diffusers(cn_model=controlnet.control_model_wrapped, lora_path=lora_path)
        # apply patches to wrapped control_model
        controlnet.control_model_wrapped.add_patches(lora_data, strength_patch=cn_lora_strength)
        # all done
        return (controlnet,)
