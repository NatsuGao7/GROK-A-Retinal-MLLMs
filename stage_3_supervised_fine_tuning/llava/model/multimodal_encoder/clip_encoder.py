import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from RETFound_MAE.models_vit import RETFound_mae
from retrieval.modeling.model import CLIPRModel
from types import SimpleNamespace
from torchvision import transforms


class CustomImageProcessor:
    def __init__(self, transforms_callable, image_mean):
        self.transforms = transforms_callable
        self.image_mean = image_mean

    def preprocess(self, images, return_tensors=None):
        # 模仿Hugging Face processor的行为，确保能处理单个或多个图像
        if not isinstance(images, list):
            images = [images]

        processed_images = [self.transforms(image) for image in images]

        # 模仿Hugging Face processor的输出格式，返回一个字典
        if return_tensors == 'pt':
            return {"pixel_values": torch.stack(processed_images)}
        else:
            return {"pixel_values": processed_images}


class OCT_2DVisionTower(nn.Module):
    def __init__(self, oct_2d_tower_path, args, delay_load=False):
        super().__init__()

        self.oct_2d_tower_weight = oct_2d_tower_path
        self.is_loaded = False

        self.custom_config = SimpleNamespace(
            image_size=224,
            patch_size=16,
            hidden_size=1024
        )

        IMG_MEAN = (0.485, 0.456, 0.406)
        IMG_STD = (0.229, 0.224, 0.225)

        _transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        self.image_processor = CustomImageProcessor(transforms_callable=_transforms, image_mean=IMG_MEAN)

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.oct_2d_tower_weight} is already loaded, skipping.')
            return

        self.oct_2d_tower = RETFound_mae(global_pool=True)

        if self.oct_2d_tower_weight:
            print(f"正在从 {self.oct_2d_tower_weight} 加载 RETFound 权重")
            checkpoint = torch.load(self.oct_2d_tower_weight, map_location='cpu', weights_only=False)
            # print(checkpoint.keys() if checkpoint.keys() is not None else "Checkpoint has no keys.")

        # # 安全地获取 'model' 键，或者直接使用整个检查点字典
        # state_dict = checkpoint.get('model2d', checkpoint)

        # # --- **核心修正代码** --- 3d-2d
        # # 创建一个新的字典，并从每个键中移除 'vit.' 这个前缀
        # new_state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items()}
        # # --- 修正结束 --- 3d-2d对齐
        # 2. 创建一个新的字典，并从每个键中移除 'backbone.' 这个前缀
        # new_state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}

        print("正在加载模型权重...")
        # 加载修正后的权重字典
        loading_report = self.oct_2d_tower.load_state_dict(checkpoint['model'], strict=False)

        print("权重加载完成。加载报告如下：")
        print(loading_report)

        self.oct_2d_tower.eval()
        self.oct_2d_tower.requires_grad_(False)
        self.is_loaded = True
        print("RETFound-OCT 模型加载完毕。")

    @torch.no_grad()
    def forward(self, images):
        image_features = self.oct_2d_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
        return image_features

    @property
    def config(self):
        return self.custom_config

    @property
    def dtype(self):
        return next(self.oct_2d_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.oct_2d_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CFP_VisionTower(nn.Module):
    def __init__(self, cfp_tower, args, delay_load=False):
        super().__init__()

        self.cfp_tower_weight = cfp_tower  # 预训练模型的路径
        self.is_loaded = False
        self.custom_config = SimpleNamespace(
            image_size=224,  # 例如: 您的模型输入是 224x224
            patch_size=14,  # 例如: ViT-L/14 的 patch size 是 14
            hidden_size=1024  # 例如: ViT-L 的隐藏层维度是 1024
        )
        IMG_MEAN = (0.485, 0.456, 0.406)
        IMG_STD = (0.229, 0.224, 0.225)

        # 1. 先定义好底层的 torchvision 转换流程
        _transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        # 2. 用我们自定义的包装器将它包装起来
        self.image_processor = CustomImageProcessor(transforms_callable=_transforms, image_mean=IMG_MEAN)

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.cfp_tower_weight))
            return

        if self.cfp_tower_weight:
            print(f"正在从 {self.cfp_tower_weight} 加载 完整的retizero 模型")
            self.model = CLIPRModel(vision_type='lora', from_checkpoint=True, weights_path=self.cfp_tower_weight,
                                    projection=False, norm_features=False, R=8)
            self.cfp_tower = self.model.vision_model.model

        self.cfp_tower.eval()
        self.cfp_tower.requires_grad_(False)
        self.is_loaded = True
        print("RetiZero-CFP 的视觉编码器部分加载完毕.")

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.cfp_tower(image.to(device=self.device, dtype=self.dtype))
                image_features.append(image_forward_out)
        else:

            image_features = self.cfp_tower(images.to(device=self.device, dtype=self.dtype))
        return image_features

    @property
    def config(self):
        return self.custom_config

    @property
    def dtype(self):
        return next(self.cfp_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.cfp_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                'Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                               output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0),
                                                        img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales,
                                                     max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
