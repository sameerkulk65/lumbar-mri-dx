"""
src/xai.py - Explainability module
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console

sys.path.insert(0, '.')
from src.model import build_model

console = Console()


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, inp, out):
            self.activations = out.detach()
        def bwd(module, gin, gout):
            self.gradients = gout[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, img_tensor, task='segmentation', class_idx=1):
        self.model.eval()
        img_tensor = img_tensor.clone().requires_grad_(True)
        head = task if task in ('segmentation', 'classification') else 'detection'
        output = self.model(img_tensor, heads={head})
        self.model.zero_grad()
        if task == 'segmentation':
            score = output['segmentation'][0, class_idx].sum()
        elif task == 'classification':
            score = output['classification'][0, :, class_idx].sum()
        else:
            score = output['detection'][0]['cls_logits'][0, class_idx].sum()
        score.backward()
        grads    = self.gradients
        acts     = self.activations
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        alpha    = grads_sq / (2 * grads_sq + sum_acts * grads_cu + 1e-8)
        weights  = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam      = torch.relu((weights * acts).sum(dim=1)).squeeze().cpu().numpy()
        cam      = cam - cam.min()
        cam      = cam / (cam.max() + 1e-8)
        import cv2
        cam = cv2.resize(cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
        return cam.astype(np.float32)


def overlay_heatmap(img_tensor, cam, alpha=0.45):
    import cv2
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img_np  = img_tensor[0].detach().cpu().numpy()
    img_np  = ((img_np * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)


def mc_dropout_uncertainty(model, img_tensor, n_passes=20):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            out = model(img_tensor, heads={'classification'})
            preds.append(out['classification'].softmax(dim=-1))
    model.eval()
    stack = torch.stack(preds, dim=0)
    mean  = stack.mean(0)
    std   = stack.std(0)
    return {
        'mean':          mean,
        'std':           std,
        'uncertain_mask': (std.max(-1).values > 0.15),
        'n_passes':      n_passes,
    }


def generate_xai_report(model, img_tensor,
                         save_dir='outputs/xai',
                         sample_id='sample_000',
                         n_mc_passes=20):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    console.print('[cyan]Generating XAI report for {}[/cyan]'.format(sample_id))

    target_layer = list(model.encoder.backbone.encoder.children())[-1]

    console.print('  GradCAM segmentation...')
    gcam_seg    = GradCAMPlusPlus(model, target_layer)
    cam_seg     = gcam_seg.generate(img_tensor, task='segmentation', class_idx=1)
    overlay_seg = overlay_heatmap(img_tensor, cam_seg)

    target_layer2 = list(model.encoder.backbone.encoder.children())[-1]
    console.print('  GradCAM classification...')
    gcam_cls    = GradCAMPlusPlus(model, target_layer2)
    cam_cls     = gcam_cls.generate(img_tensor, task='classification', class_idx=2)
    overlay_cls = overlay_heatmap(img_tensor, cam_cls)

    console.print('  MC-Dropout ({} passes)...'.format(n_mc_passes))
    unc = mc_dropout_uncertainty(model, img_tensor, n_mc_passes)

    img_np = (img_tensor[0, 0].detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('XAI Report - {}'.format(sample_id), fontsize=13)

    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input MRI')
    axes[0].axis('off')

    axes[1].imshow(overlay_seg)
    axes[1].set_title('GradCAM - Segmentation')
    axes[1].axis('off')

    axes[2].imshow(overlay_cls)
    axes[2].set_title('GradCAM - Classification')
    axes[2].axis('off')

    unc_reshaped = unc['std'][0].mean(-1).cpu().numpy().reshape(5, 5)
    im = axes[3].imshow(unc_reshaped, cmap='Reds', aspect='auto', vmin=0, vmax=0.3)
    axes[3].set_xticks(range(5))
    axes[3].set_xticklabels(['L1/L2','L2/L3','L3/L4','L4/L5','L5/S1'], fontsize=8)
    axes[3].set_yticks(range(5))
    axes[3].set_yticklabels(['SCS','LFN','RFN','LSS','RSS'], fontsize=8)
    axes[3].set_title('MC-Dropout Uncertainty')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = save_path / '{}_xai_report.png'.format(sample_id)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()

    console.print('[green]Saved:[/green] {}'.format(out_path))
    console.print('  Mean uncertainty : {:.4f}'.format(unc['std'].mean().item()))
    console.print('  Uncertain tasks  : {}/25'.format(int(unc['uncertain_mask'].sum().item())))
    return {'cam_seg': cam_seg, 'cam_cls': cam_cls, 'uncertainty': unc, 'report_path': str(out_path)}


if __name__ == '__main__':
    import yaml
    console.rule('[bold cyan]Step 7 - XAI Module[/bold cyan]')
    with open('configs/config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    console.print('Building model...')
    model = build_model(cfg)
    model.eval()

    img_tensor = torch.randn(1, 3, 512, 512)

    console.rule('GradCAM test')
    target_layer = list(model.encoder.backbone.encoder.children())[-1]
    gcam = GradCAMPlusPlus(model, target_layer)
    cam  = gcam.generate(img_tensor, task='segmentation', class_idx=1)
    console.print('CAM shape : {}'.format(cam.shape))
    console.print('CAM range : [{:.3f}, {:.3f}]'.format(cam.min(), cam.max()))

    overlay = overlay_heatmap(img_tensor, cam)
    console.print('Overlay   : {}'.format(overlay.shape))

    console.rule('MC-Dropout test')
    unc = mc_dropout_uncertainty(model, img_tensor, n_passes=10)
    console.print('Mean shape : {}'.format(list(unc['mean'].shape)))
    console.print('Std  shape : {}'.format(list(unc['std'].shape)))
    console.print('Avg unc    : {:.4f}'.format(unc['std'].mean().item()))
    console.print('Uncertain  : {}/25'.format(int(unc['uncertain_mask'].sum().item())))

    console.rule('Full XAI report')
    result = generate_xai_report(model, img_tensor,
                                  save_dir='outputs/xai',
                                  sample_id='test_sample',
                                  n_mc_passes=10)

    console.print('')
    console.print('[bold green]Step 7 COMPLETE.[/bold green]')
    console.print('Next: python src/deploy.py')
