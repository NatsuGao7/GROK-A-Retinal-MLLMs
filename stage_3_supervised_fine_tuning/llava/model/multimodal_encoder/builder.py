import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, OCT_2DVisionTower, CFP_VisionTower
import torch

def build_oct_2d_tower(oct_2d_tower_cfg, **kwargs):
    # 模仿格式
    checkpoint_path = getattr(oct_2d_tower_cfg, 'mm_oct_2d_tower', getattr(oct_2d_tower_cfg, 'oct_2d_tower', None))
    is_absolute_path_exists = os.path.exists(checkpoint_path)
    if is_absolute_path_exists: #只要给权重路径，就实例化2D-OCT编码器
        return OCT_2DVisionTower(checkpoint_path, args=oct_2d_tower_cfg, **kwargs) 

    raise ValueError(f'Unknown oct_2d tower: {checkpoint_path}')

def build_cfp_tower(cfp_tower_cfg, **kwargs):
    # 模仿格式
    checkpoint_path = getattr(cfp_tower_cfg, 'mm_cfp_tower', getattr(cfp_tower_cfg, 'cfp_tower', None))
    is_absolute_path_exists = os.path.exists(checkpoint_path)
    if is_absolute_path_exists: #只要给权重路径，就实例化 CFP 编码器
        return CFP_VisionTower(checkpoint_path, args=cfp_tower_cfg, **kwargs) 

    raise ValueError(f'Unknown cfp tower: {checkpoint_path}')


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CFP_VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


