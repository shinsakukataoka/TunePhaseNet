#!/usr/bin/env python
import argparse, os, logging, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc
from tqdm import tqdm
from scipy.signal import find_peaks


# ------------------------- helpers ------------------------- #
def get_phase_dict(dataset_name: str):
    base = {
        "trace_p_arrival_sample": "P", "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P", "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P", "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P", "trace_pwPm_arrival_sample": "P",
        "trace_s_arrival_sample": "S", "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S", "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S", "trace_Sn_arrival_sample": "S",
    }
    if dataset_name == "STEAD":
        base.update({"trace_P_arrival_sample": "P", "trace_S_arrival_sample": "S"})
    return base


# ------------------------- main ------------------------- #
def main(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Running baseline on {device}")

    # 1) model
    model = sbm.PhaseNet.from_pretrained(args.pretrained_model).to(device).eval()
    logging.info(f"Loaded pre-trained weights: {args.pretrained_model}")

    # 2) data
    if args.dataset_name == "Iquique":
        data = sbd.Iquique(sampling_rate=100, force_download=args.force_download_data)
    elif args.dataset_name == "STEAD":
        data = sbd.STEAD(sampling_rate=100, component_order="ZNE",
                         force_download=args.force_download_data)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset_name}")
    _, _, test_view = data.train_dev_test()
    phase_dict = get_phase_dict(args.dataset_name)
    aug = [
        sbg.WindowAroundSample(list(phase_dict.keys()),
                               samples_before=args.context_window_samples_before,
                               windowlen=args.context_window_length,
                               selection="random", strategy="variable"),
        sbg.RandomWindow(windowlen=args.model_input_samples, strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(label_columns=phase_dict,
                                  model_labels=model.labels,
                                  sigma=args.sigma, dim=0),
        sbg.ChangeDtype(np.float32),
    ]
    gen = sbg.GenericGenerator(test_view); gen.add_augmentations(aug)
    loader = DataLoader(gen, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, worker_init_fn=worker_seeding,
                        pin_memory=True)
    logging.info(f"Test traces: {len(gen)}")

    # 3) inference
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            x = batch["X"].to(device)
            preds.append(model(x).cpu().numpy())
            trues.append(batch["y"].cpu().numpy())
    y_pred = np.concatenate(preds); y_true = np.concatenate(trues)

    # 4) metrics
    records, plots = [], []
    for idx, phase in enumerate(model.labels):
        if phase == "N" and args.skip_noise_metrics:
            continue
        t_flat = y_true[:, :, idx].ravel()
        p_flat = y_pred[:, :, idx].ravel()
        y_true_bin = (t_flat >= 0.5).astype(int)
        prec, rec, _ = precision_recall_curve(y_true_bin, p_flat)
        pr_auc = auc(rec, prec)
        y_pred_bin = (p_flat >= args.detection_threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="binary", zero_division=0)
        records.append(dict(Phase=phase, Precision=p, Recall=r,
                            F1=f1, PR_AUC=pr_auc,
                            Threshold=args.detection_threshold))
        plots.append((rec, prec, phase, pr_auc, p, r))

    metrics_df = pd.DataFrame(records)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir,
                            f"{args.pretrained_model}_{args.dataset_name}"
                            f"_metrics_thresh{args.detection_threshold}.csv")
    metrics_df.to_csv(csv_path, index=False)
    logging.info(f"Saved metrics → {csv_path}")
    logging.info("\n" + metrics_df.to_string(index=False))

    # 5) PR-curve figure
    nrows = len(plots)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 4 * nrows))
    if nrows == 1:
        axes = [axes]
    for ax, (rec, prec, ph, pr_auc, p_op, r_op) in zip(axes, plots):
        ax.plot(rec, prec, lw=2, label=f"PR curve (AUC={pr_auc:.3f})")
        ax.plot(r_op, p_op, "ro",
                label=f"@{args.detection_threshold}: P={p_op:.3f}, R={r_op:.3f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{ph} phase"); ax.legend(); ax.grid(True)
    fig.tight_layout()
    pr_path = os.path.join(args.output_dir,
                           f"{args.pretrained_model}_{args.dataset_name}_PR.png")
    fig.savefig(pr_path); plt.close(fig)
    logging.info(f"Saved PR curves → {pr_path}")


# ------------------------- argparse ------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser("PhaseNet baseline evaluation")
    ap.add_argument("--dataset_name", default="Iquique", choices=["Iquique", "STEAD"])
    ap.add_argument("--pretrained_model", default="stead",
                    choices=["stead", "instance", "ethz", "geofon"])
    ap.add_argument("--force_download_data", action="store_true")
    ap.add_argument("--context_window_samples_before", type=int, default=3000)
    ap.add_argument("--context_window_length", type=int, default=6000)
    ap.add_argument("--model_input_samples", type=int, default=3001)
    ap.add_argument("--sigma", type=float, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--detection_threshold", type=float, default=0.3)
    ap.add_argument("--skip_noise_metrics", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_dir", default="baseline_results")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if GPU is available")
    args = ap.parse_args()
    main(args)

