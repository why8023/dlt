import numpy as np
from pathlib import Path
from lg_template import ECGPPG
import torch
from tqdm import tqdm

# test
model = ECGPPG.load_from_checkpoint(r"best\epoch=044-val=0.476871.ckpt")
model.freeze()
model.eval()

test_path = [
    Path(
        r"\\192.168.2.8\xjk\Algorithm\PPG_AF\clinic_Phase2_analysis\WECGMatchWPPGAlign_Sample_Invalid"
    ),
    Path(
        r"\\192.168.2.8\xjk\Algorithm\PPG_AF\clinic_Phase2_analysis\WECGMatchWPPGAlign_Sample"
    ),
]
for p in test_path:
    files = list(p.glob("**/patch_cnn_feauture.npy"))
    for f in tqdm(files):
        print(f)
        sample_map = np.load(f)
        sample_map = sample_map[:, np.newaxis, :]
        with torch.no_grad():
            preds = model(torch.from_numpy(sample_map).float().to("cuda"))
            preds = preds.to("cpu").numpy()
            np.save(f.parent / f.name.replace("patch_cnn_feauture", "pred_ppg_feature"), preds)
            print(f"save to {f.parent / f.name.replace('patch_cnn_feauture', 'pred_ppg_feature')}")
