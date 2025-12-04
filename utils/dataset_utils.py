import os
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from utils.action_utils import YAW_LIST, PITCH_LIST


def angle_str(angle: float) -> str:
    s = f"{angle:.4f}"
    return s.rstrip("0").rstrip(".")


class ShapeNetDataset(Dataset):
    def __init__(self, root: str,
                 split: str = "train",
                 max_objs_per_synset: Optional[int] = None,
                 img_size: int = 256):
        super().__init__()
        self.root = root
        self.split = split

        trans = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ]
        self.transform = transforms.Compose(trans)

        all_synsets = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
        self.synsets = all_synsets

        self.obj_sequences: List[Dict[str, Any]] = []
        self._build_index(max_objs_per_synset=max_objs_per_synset)

    def _build_index(self, max_objs_per_synset: Optional[int]):
        for syn in self.synsets:
            syn_dir = os.path.join(self.root, syn, self.split)
            if not os.path.isdir(syn_dir):
                continue

            obj_ids = sorted(
                d for d in os.listdir(syn_dir)
                if os.path.isdir(os.path.join(syn_dir, d))
            )
            if max_objs_per_synset is not None:
                obj_ids = obj_ids[:max_objs_per_synset]

            print(f"[SeqImageDataset] synset={syn}, num_objs={len(obj_ids)}")

            for obj_id in obj_ids:
                gen_dir = os.path.join(syn_dir, obj_id, "generated")
                if not os.path.isdir(gen_dir):
                    print(f"[WARN] {syn}/{obj_id} has no generated/ dir, skip")
                    continue

                img_paths: List[str] = []
                missing = False
                for yaw in YAW_LIST:
                    y_str = angle_str(yaw)
                    for pitch in PITCH_LIST:
                        p_str = angle_str(pitch)
                        fname = f"view_y{y_str}_p{p_str}.png"
                        path = os.path.join(gen_dir, fname)
                        if not os.path.isfile(path):
                            print(f"[MISSING] {syn}/{obj_id} missing {fname}")
                            missing = True
                            break
                        img_paths.append(path)
                    if missing:
                        break

                if missing or len(img_paths) == 0:
                    print(f"[SKIP] {syn}/{obj_id} due to missing views")
                    continue

                self.obj_sequences.append(
                    dict(
                        synset=syn,
                        obj_id=obj_id,
                        img_paths=img_paths,
                        seq_len=len(img_paths),
                    )
                )

        print(f"[SeqImageDataset] total valid objects = {len(self.obj_sequences)}")

    def __len__(self) -> int:
        return len(self.obj_sequences)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        rec = self.obj_sequences[idx]
        paths = rec["img_paths"]
        T = rec["seq_len"]

        imgs = [self._load_image(p) for p in paths]
        images = torch.stack(imgs, dim=0)

        meta = {
            "synset": rec["synset"],
            "obj_id": rec["obj_id"],
            "seq_len": T,
        }
        return images, meta

