"""Compute CLIP text embedding cosine similarity matrices for curriculum sampler."""
import sys
sys.path.insert(0, "/home/avoidman2233/Desktop/LVLM/ATPrompt")

import torch
import clip
from dassl.config import get_cfg_default
import train as _t

DATA = "/home/avoidman2233/Desktop/LVLM/DATA"
OUTPUT_DIR = "/home/avoidman2233/Desktop/LVLM/ATPrompt/output/curriculum"

DATASETS = {
    "stanford_cars": "StanfordCars",
    "eurosat": "EuroSAT",
    "dtd": "DescribableTextures",
}

TEMPLATES = {
    "StanfordCars": "a photo of a {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "DescribableTextures": "{} texture.",
}

def get_base_class_names(dataset_name):
    """Get base class names using Dassl data manager."""
    import datasets.stanford_cars, datasets.eurosat, datasets.dtd
    cfg = get_cfg_default()
    _t.extend_cfg(cfg)
    cfg.DATASET.ROOT = DATA
    cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.INPUT.SIZE = (224, 224)
    cfg.INPUT.TRANSFORMS = ["random_resized_crop", "random_flip", "normalize"]
    cfg.merge_from_file(f"configs/datasets/{dataset_name}.yaml")
    
    from dassl.data import DataManager
    dm = DataManager(cfg)
    
    # Get class names from lab2cname
    class_names = []
    for label in range(dm.num_classes):
        class_names.append(dm.lab2cname[label])
    
    return class_names

def compute_similarity(class_names, template):
    from clip.clip import _MODELS, _download
    backbone_name = "ViT-B/16"
    model_path = _download(_MODELS[backbone_name])
    
    try:
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    
    design_details = {"trainer": "CoOp", "vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0}
    model = clip.build_model(state_dict, design_details)
    model.eval()
    
    prompts = [template.format(name) for name in class_names]
    tokens = clip.tokenize(prompts)
    
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        sim = features @ features.T
    
    return sim

if __name__ == "__main__":
    for ds_name, cls_name in DATASETS.items():
        print(f"\n=== {ds_name} ({cls_name}) ===")
        names = get_base_class_names(ds_name)
        print(f"  Base classes: {len(names)}")
        print(f"  Sample: {names[:3]}")
        
        template = TEMPLATES[cls_name]
        sim = compute_similarity(names, template)
        
        out_path = f"{OUTPUT_DIR}/{ds_name}_sim.pt"
        torch.save(sim, out_path)
        
        print(f"  Matrix shape: {sim.shape}")
        print(f"  Diagonal (first 3): {sim.diag()[:3]}")
        print(f"  Min off-diagonal: {sim[sim < 0.999].min():.4f}")
        print(f"  Max off-diagonal: {sim[sim < 0.999].max():.4f}")
        print(f"  Saved to: {out_path}")
    
    print("\nDone!")
