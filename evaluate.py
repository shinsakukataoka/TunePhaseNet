#!/usr/bin/env python
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
from sklearn.metrics import precision_recall_curve
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc
import argparse
import os
import pandas as pd
from tqdm import tqdm
import logging
from scipy.signal import find_peaks
import glob
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_phase_dict(dataset_name):
    base_phase_dict = {
        "trace_p_arrival_sample": "P", "trace_P_arrival_sample": "P",
        "trace_P1_arrival_sample": "P", "trace_Pg_arrival_sample": "P",
        "trace_Pn_arrival_sample": "P", "trace_PmP_arrival_sample": "P",
        "trace_pwP_arrival_sample": "P", "trace_pwPm_arrival_sample": "P",
        "trace_s_arrival_sample": "S", "trace_S_arrival_sample": "S",
        "trace_S1_arrival_sample": "S", "trace_Sg_arrival_sample": "S",
        "trace_SmS_arrival_sample": "S", "trace_Sn_arrival_sample": "S",
    }
    if dataset_name == "STEAD":
        base_phase_dict.update({
            "trace_P_arrival_sample": "P",
            "trace_S_arrival_sample": "S",
        })
    return base_phase_dict

def extract_discrete_picks(prob_trace, detection_threshold, pick_prominence=0.1, pick_distance_samples=50):
    picks_indices, _ = find_peaks(
        prob_trace,
        height=detection_threshold,
        prominence=pick_prominence,
        distance=pick_distance_samples
    )
    return picks_indices


def main(args):
    logging.info(f"Running evaluation with arguments: {args}")

    model = sbm.PhaseNet(phases="PSN", norm="peak") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)

    if not os.path.exists(args.model_path):
        logging.error(f"Model path {args.model_path} does not exist. Exiting.")
        return
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Successfully loaded model from: {args.model_path}")
    except Exception as e:
        logging.error(f"Error loading model state_dict from {args.model_path}: {e}.")
        return
    model.eval()

    logging.info(f"Loading TEST set for dataset: {args.dataset_name}")
    try:
        if args.dataset_name == "Iquique":
            data = sbd.Iquique(sampling_rate=100, force_download=args.force_download_data)
        elif args.dataset_name == "STEAD":
            data = sbd.STEAD(sampling_rate=100, force_download=args.force_download_data, component_order="ZNE")
        else:
            raise ValueError(f"Unknown or unsupported dataset: {args.dataset_name}")
        
        if hasattr(data, 'citations') and data.citations:
             logging.info(f"Dataset citations: {data.citations}")

        train_data_view, dev_data_view, test_data_view = data.train_dev_test()
        dev_generator = sbg.GenericGenerator(dev_data_view)
        logging.info(f"Test set size (using default split): {len(test_data_view)}")
    except Exception as e:
        logging.error(f"Error loading or splitting dataset {args.dataset_name} for evaluation: {e}")
        return
    
    phase_dict = get_phase_dict(args.dataset_name)
    
    test_augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=args.context_window_samples_before,
            windowlen=args.context_window_length,
            selection="random",
            strategy="variable"
        ),
        sbg.RandomWindow(
            windowlen=args.model_input_samples,
            strategy="pad"
        ),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict,
            model_labels=model.labels,
            sigma=args.sigma,
            dim=0
        ),
        sbg.ChangeDtype(np.float32)
    ]
    dev_generator  = sbg.GenericGenerator(dev_data_view)
    dev_generator.add_augmentations(test_augmentations)
    test_generator = sbg.GenericGenerator(test_data_view)
    test_generator.add_augmentations(test_augmentations)

    if len(test_generator) == 0:
        logging.error("Test generator is empty. Check data and augmentations.")
        return

    test_loader = DataLoader(
        test_generator, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_seeding, pin_memory=True
    )

    all_preds_prob, all_labels_prob, all_trace_ids_for_plotting = [], [], []
    logging.info("Running evaluation on the test set...")
    if len(test_loader) == 0:
        logging.error("Test loader is empty. Cannot perform evaluation.")
        return
        
    with torch.no_grad():
        for batch_id_val, batch in enumerate(tqdm(test_loader, desc="Evaluating Test Set")):
            X_batch = batch["X"].to(device)
            y_batch_prob = batch["y"].to(device).float() 

            if batch_id_val == 0:
                 logging.info(f"First evaluation batch X shape: {X_batch.shape} (dtype: {X_batch.dtype}), y shape: {y_batch_prob.shape} (dtype: {y_batch_prob.dtype})")
                 if X_batch.shape[-1] != args.model_input_samples:
                    logging.error(f"CRITICAL SHAPE MISMATCH IN EVAL: X_batch last dim {X_batch.shape[-1]} != model_input_samples {args.model_input_samples}")
                    return 
                 if y_batch_prob.dtype != torch.float32 or X_batch.dtype != torch.float32:
                     logging.warning(f"Expected float32 tensors. Got X: {X_batch.dtype}, y: {y_batch_prob.dtype}")

            pred_prob = model(X_batch).cpu().numpy() 
            all_preds_prob.append(pred_prob)
            all_labels_prob.append(y_batch_prob.cpu().numpy()) 
            if "id" in batch: all_trace_ids_for_plotting.extend(batch["id"])
            elif "trace_name" in batch: all_trace_ids_for_plotting.extend(batch["trace_name"])

    if not all_preds_prob:
        logging.error("No predictions were generated. Aborting metric calculation and plotting.")
        return

    y_pred_prob_all = np.concatenate(all_preds_prob, axis=0)
    y_true_prob_all = np.concatenate(all_labels_prob, axis=0) 
    
    report_data = []
    num_phases_to_plot_pr = len(model.labels) -1 if ('N' in model.labels and args.skip_noise_metrics) else len(model.labels)
    if num_phases_to_plot_pr <=0: num_phases_to_plot_pr=1 
        
    fig_pr, axs_pr = plt.subplots(num_phases_to_plot_pr, 1, 
                                 figsize=(8, 5 * num_phases_to_plot_pr), squeeze=False)
    axs_pr_flat = axs_pr.flatten()
    plot_idx_pr = 0

    for i, phase_name in enumerate(model.labels):
        if phase_name == 'N' and args.skip_noise_metrics: continue
        true_phase_probs = y_true_prob_all[:, :, i].flatten()
        pred_phase_probs = y_pred_prob_all[:, :, i].flatten()
        #precision_vals, recall_vals, _ = precision_recall_curve(true_phase_probs, pred_phase_probs)
        y_true_binary = (true_phase_probs >= 0.5).astype(int)
        precision_vals, recall_vals, _ = precision_recall_curve(
            y_true_binary,pred_phase_probs
        )
        pr_auc_score = auc(recall_vals, precision_vals)
        y_pred_binary = (pred_phase_probs >= args.detection_threshold).astype(int)
        y_true_binary = (true_phase_probs >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        report_data.append({"Phase": phase_name, "Precision": p, "Recall": r, "F1-Score": f1, "PR_AUC": pr_auc_score, "Threshold_for_PRF1": args.detection_threshold})
        ax = axs_pr_flat[plot_idx_pr]
        ax.plot(recall_vals, precision_vals, lw=2, label=f'PR curve (area = {pr_auc_score:.3f})')
        ax.plot(r, p, 'ro', markersize=8, label=f'Op. point (Thresh={args.detection_threshold:.2f})\nP={p:.3f}, R={r:.3f}')
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f'PR Curve: {phase_name}'); ax.legend(loc="best"); ax.grid(True)
        plot_idx_pr += 1

    metrics_df = pd.DataFrame(report_data)
    logging.info("\n--- Evaluation Metrics --- \n" + metrics_df.to_string())
    os.makedirs(args.output_dir, exist_ok=True)
    base_model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    metrics_path = os.path.join(args.output_dir, f"{base_model_name}_{args.dataset_name}_metrics_thresh{args.detection_threshold}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"Saved metrics to {metrics_path}")

    if plot_idx_pr > 0:
        pr_curve_path = os.path.join(args.output_dir, f"{base_model_name}_{args.dataset_name}_PR_curves.png")
        fig_pr.tight_layout(); fig_pr.savefig(pr_curve_path)
        logging.info(f"Saved PR curves to {pr_curve_path}")
    plt.close(fig_pr)

    sample = dev_generator[np.random.randint(len(dev_generator))]
    X, y = sample["X"], sample["y"]
    with torch.no_grad():
        pred = model(torch.tensor(X, device=device).unsqueeze(0))[0].cpu().numpy()

    # re-plot exactly as in training script
    fig = plt.figure(figsize=(15,10))
    axs = fig.subplots(3, 1, sharex=True, gridspec_kw={"hspace":0, "height_ratios":[3,1,1]})
    axs[0].plot(X.T); axs[0].set_title("Input")
    axs[1].plot(y.T); axs[1].set_title("Ground Truth Prob.")
    axs[2].plot(pred.T); axs[2].set_title("Model Prediction")
    axs[2].set_xlabel("Time Samples")
    pd.DataFrame(X).to_csv(os.path.join(args.output_dir, "example_input.csv"), index=False)
    pd.DataFrame(y).to_csv(os.path.join(args.output_dir, "example_ground_truth.csv"), index=False)
    pd.DataFrame(pred).to_csv(os.path.join(args.output_dir, "example_prediction.csv"), index=False)
    plt.savefig(os.path.join(args.output_dir, "seismic_phase_prediction.png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PhaseNet model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained PhaseNet model (.pth).")
    parser.add_argument("--dataset_name", type=str, default="Iquique", help="SeisBench dataset name.")
    parser.add_argument("--force_download_data", action="store_true", help="Force download dataset.")
    parser.add_argument("--context_window_samples_before", type=int, default=3000, help="Context window samples before pick.")
    parser.add_argument("--context_window_length", type=int, default=6000, help="Context window length.")
    parser.add_argument("--model_input_samples", type=int, default=3001, help="The exact number of samples PhaseNet model expects.")
    parser.add_argument("--sigma", type=float, default=20, help="Sigma for ProbabilisticLabeller (should match training).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--detection_threshold", type=float, default=0.3, help="Threshold for PRF1 & discrete picks.")
    parser.add_argument("--skip_noise_metrics", action="store_true", help="Skip 'N' (Noise) phase evaluation.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_phasenet", help="Dir for evaluation outputs.")
    parser.add_argument("--num_plot_examples", type=int, default=5, help="Number of example traces to plot.")
    
    args = parser.parse_args()
    main(args)