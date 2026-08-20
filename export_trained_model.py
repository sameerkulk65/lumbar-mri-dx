"""
Load a state_dict checkpoint (e.g. trained in Colab, saved via
src.model.save_checkpoint) and export it as the TorchScript file
the Streamlit app / ClinicalInferenceEngine actually loads.

Usage:
    python export_trained_model.py path/to/checkpoint.pt
"""
import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, ".")
from src.model import build_model
from src.deploy import export_torchscript

def main():
    if len(sys.argv) != 2:
        print("Usage: python export_trained_model.py <checkpoint.pt>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    full_state = ckpt.get("state_dict", ckpt)
    # Lightning module wraps the architecture as `self.model` and also
    # carries a sibling `self.loss_fn` (uncertainty-weighting params) in
    # the same state_dict — keep only the `model.*` keys.
    state = {
        k[len("model."):]: v
        for k, v in full_state.items()
        if k.startswith("model.")
    }
    # Checkpoints trained before the morph_head was added won't have those
    # keys -- load them with strict=False in that case (morph_head stays
    # randomly initialized) instead of failing the whole export.
    has_morph = any(k.startswith("morph_head.") for k in state)
    missing, unexpected = model.load_state_dict(state, strict=has_morph)
    if unexpected:
        print(f"WARNING: unexpected keys in checkpoint: {unexpected}")
    if not has_morph:
        print("NOTE: checkpoint predates the disc-morphology head -- "
              "morph_head is randomly initialized (untrained). Its "
              "predictions in the UI will be meaningless until you retrain "
              "with the Sudirman dataset and re-export.")
    print(f"Loaded weights from epoch {ckpt.get('epoch', '?')}")
    model.eval()

    # Plain state_dict for the app's eager-mode XAI model (GradCAM/uncertainty
    # need autograd and dropout-in-train-mode, which the traced export can't do).
    state_dict_path = "outputs/exports/lumbar_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    print(f"Trained state_dict saved to {state_dict_path}")

    out_path = export_torchscript(model, "outputs/exports/lumbar_model.pt", cfg["data"]["image_size"])
    print(f"Trained model exported to {out_path} — restart the Streamlit app to pick it up.")

if __name__ == "__main__":
    main()
