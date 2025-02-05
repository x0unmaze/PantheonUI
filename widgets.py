import ipywidgets.widgets as w
from constants import PREPROCESSORS, CONTROLNET_LINKS, SAMPLER_NAMES, SCHEDULER_NAMES

class ConfigInput:
    def __init__(self):
        self.cfg = w.FloatText(description='cfg',value=7.0,min=2,max=30,step=0.1,layout={'width':'inherit'})
        self.steps = w.IntText(description='steps',value=30,min=8,max=60,step=1,layout={'width':'inherit'})
        self.seed = w.IntText(description='seed',value=0,layout={'width':'inherit'})
        self.sampler = w.Dropdown(description='sampler',options=SAMPLER_NAMES,layout={'width':'inherit'})
        self.scheduler = w.Dropdown(description='scheduler',options=SCHEDULER_NAMES,layout={'width':'inherit'})
        self.width = w.IntSlider(description='width',value=768,min=512,max=1024,step=64,layout={'width':'inherit'})
        self.height = w.IntSlider(description='height',value=768,min=512,max=1024,step=64,layout={'width':'inherit'})
        self.num_image_per_prompt = w.IntSlider(description='num_image_per_prompt',value=1,min=1,max=6,step=1,layout={'width':'inherit'})
        self.denoising = w.FloatSlider(description='denoising',value=1,min=0.0,max=1.0,step=0.01,layout={'width':'inherit'})
        self.hires_fix = w.Checkbox(description='hires_fix',layout={'width':'inherit'})
    
    def values(self):
        return {
            'cfg': self.cfg.value,
            'steps': self.steps.value,
            'seed': self.seed.value,
            'sampler': self.sampler.value,
            'scheduler': self.scheduler.value,
            'width': self.width.value,
            'height': self.height.value,
            'num_image_per_prompt': self.num_image_per_prompt.value,
            'denoising': self.denoising.value,
            'hires_fix': self.hires_fix.value,
        }

    def widget(self, layout={}):
        return w.VBox([
            self.sampler,
            self.scheduler,
            self.cfg,
            self.steps,
            self.seed,
            self.width,
            self.height,
            self.num_image_per_prompt,
            self.denoising,
            self.hires_fix,
        ], layout=layout)

class ControlnetInput:
    def __init__(self):
        self.enable = w.Checkbox(description='enable',value=False,layout={'width':'inherit'})
        self.base_image = w.Text(description='base_image', placeholder='/content/base.jpg',layout={'width':'inherit'})
        self.mask_image = w.Text(description='mask_image', placeholder='/content/mask.jpg',layout={'width':'inherit'})
        self.preprocessor = w.Dropdown(description='preprocessor', options=['none']+PREPROCESSORS,layout={'width':'inherit'})
        self.model = w.Dropdown(description='model',placeholder='controlnet_depth_sd15.safetensors', options=list(CONTROLNET_LINKS.keys()),layout={'width':'inherit'})
        self.strength = w.FloatSlider(description='strength',value=1.0,min=0.0,max=1.0,step=0.01,layout={'width':'inherit'})
        self.start = w.FloatSlider(description='start',value=1.0,min=0.0,max=1.0,step=0.01,layout={'width':'inherit'})
        self.end = w.FloatSlider(description='end',value=1.0,min=0.0,max=1.0,step=0.01,layout={'width':'inherit'})

    def values(self):
        return {
            'enable': self.enable.value,
            'base_image': self.base_image.value,
            'mask_image': self.mask_image.value,
            'preprocessor': self.preprocessor.value,
            'model': self.model.value,
            'strength': self.strength.value,
            'start': self.start.value,
            'end': self.end.value,
        }

    def widget(self, layout={}):
        return w.VBox([
            self.enable,
            self.base_image,
            self.mask_image,
            self.preprocessor,
            self.model,
            self.strength,
            self.start,
            self.end,
        ], layout=layout)

class ControlnetGroupInput:
    def __init__(self, length: int = 3):
        self.groups = [ControlnetInput() for i in range(length)]
    
    def values(self):
        return [item.values() for item in self.groups]

    def widget(self, layout={}):
        return w.Tab(
            children=[cn.widget() for cn in self.groups],
            titles=[f'controlnet_{i}' for i in range(len(self.groups))],
            layout=layout,
        )

class ConditionInput:
    def __init__(self):
        self.base_image = w.Text(description='base_image', placeholder='/content/base.jpg',layout={'width':'inherit'})
        self.mask_image = w.Text(description='mask_image', placeholder='/content/mask.jpg',layout={'width':'inherit'})
        self.positive_prompt = w.Textarea(description='positive', rows=5, placeholder='positive prompt ...',layout={'width':'inherit'})
        self.negative_prompt = w.Textarea(description='negative', rows=5, placeholder='negative prompt ...',layout={'width':'inherit'})

    def values(self):
        return {
            'base_image': self.base_image.value,
            'mask_image': self.mask_image.value,
            'positive_prompt': self.positive_prompt.value,
            'negative_prompt': self.negative_prompt.value,
        }

    def widget(self, layout={}):
        return w.VBox([
            self.base_image,
            self.mask_image,
            self.positive_prompt,
            self.negative_prompt,
        ], layout=layout)

class LoraInput:
    def __init__(self):
        self.name = w.Text(description='lora_path',layout={'width':'inherit'})
        self.strength = w.FloatSlider(description=' ',value=1,min=0,max=1,step=0.01,layout={'width':'inherit'})

    def values(self):
        return {'name': self.name.value, 'strength': self.strength.value}

    def widget(self, layout={}):
        return w.VBox([
            self.name,
            self.strength,
        ], layout=layout)

class LoraGroupInput:
    def __init__(self, length=3):
        self.groups = [LoraInput() for i in range(length)]
    
    def values(self):
        return [item.values() for item in self.groups]

    def widget(self, layout={}):
        return w.VBox(
            children=[item.widget() for item in self.groups],
            layout=layout,
        )

class SimpleGenerator:
    def __init__(self, ldm_paths=[], vae_paths=[]):
        self.ldm_path = w.Combobox(description='ldm_path',options=ldm_paths,layout={'width':'50%'})
        self.vae_path = w.Combobox(description='vae_path',options=vae_paths,layout={'width':'50%'})
        self.controlnet_panel = ControlnetGroupInput()
        self.config_panel = ConfigInput()
        self.condition_panel = ConditionInput()
        self.lora_panel = LoraGroupInput()
        self.submit_btn = w.Button(description='submit', button_style='primary')
    
    def values(self):
        return {
            'ldm_path': self.ldm_path.value,
            'vae_path': self.vae_path.value,
            'condition': self.condition_panel.values(),
            'config': self.config_panel.values(),
            'controlnet': self.controlnet_panel.values(),
            'lora': self.lora_panel.values(),
        }

    def widget(self):
        wrapper = w.GridBox(
            children=[
                w.HBox([self.ldm_path, self.vae_path], layout={'grid_area': 'ckpt'}),
                self.config_panel.widget({'grid_area':'config'}),
                self.condition_panel.widget({'grid_area':'condition'}),
                self.controlnet_panel.widget({'grid_area':'cnet'}),
                self.lora_panel.widget({'grid_area': 'lora', 'border': '1px dashed gray'}),
                w.HBox([self.submit_btn], layout={'grid_area':'buttons'})
            ],
            layout=w.Layout(
                grid_gap='12px',
                grid_template_rows='auto auto auto',
                grid_template_columns='auto auto auto auto',
                grid_template_areas='''
                "config ckpt ckpt ckpt"
                "config condition condition condition"
                "lora cnet cnet cnet"
                "buttons buttons buttons buttons"
                '''
            )
        )
        return wrapper