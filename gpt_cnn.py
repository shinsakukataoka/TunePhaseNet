#!/usr/bin/env python
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import logging

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

def original_custom_loss_fn(y_pred, y_true, eps=1e-5):
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(dim=-1)  # Shape: (batch_size, num_classes)
    h = h.sum(dim=-1)   # Shape: (batch_size)
    h = h.mean()        # Shape: scalar
    return -h

def main(args):
    logging.info(f"Running training with arguments: {args}")

    if args.model_input_samples > args.context_window_length:
        logging.warning(f"model_input_samples ({args.model_input_samples}) is greater than context_window_length ({args.context_window_length}).")

    # --- Model Initialization ---
    if args.use_pretrained_weights:
        logging.info(f"Loading pre-trained PhaseNet model: {args.pretrained_model_name} for fine-tuning.")
        try:
            model = sbm.PhaseNet.from_pretrained(args.pretrained_model_name, phases="PSN", norm="peak")
        except Exception as e:
            logging.error(f"Could not load pre-trained model {args.pretrained_model_name}. Error: {e}")
            logging.info("Falling back to a new PhaseNet instance from scratch.")
            model = sbm.PhaseNet(phases="PSN", norm="peak")
    else:
        logging.info("Initializing a new PhaseNet model from scratch.")
        model = sbm.PhaseNet(phases="PSN", norm="peak")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Model moved to {device}.")

    # --- Data Loading ---
    logging.info(f"Loading dataset: {args.dataset_name}")
    try:
        if args.dataset_name == "Iquique":
            data = sbd.Iquique(sampling_rate=100, force_download=args.force_download_data)
        elif args.dataset_name == "STEAD":
            data = sbd.STEAD(sampling_rate=100, force_download=args.force_download_data, component_order="ZNE")
        else:
            raise ValueError(f"Unknown or unsupported dataset: {args.dataset_name}")
        
        if hasattr(data, 'citations') and data.citations:
            logging.info(f"Dataset citations: {data.citations}")

        train_data, dev_data, _ = data.train_dev_test()
        logging.info(f"Dataset split (using default): Train={len(train_data)}, Dev={len(dev_data)}")

    except Exception as e:
        logging.error(f"Error loading or splitting dataset {args.dataset_name}: {e}")
        return

    phase_dict = get_phase_dict(args.dataset_name)

    # --- Data Augmentation & Generation ---
    train_generator = sbg.GenericGenerator(train_data)
    dev_generator = sbg.GenericGenerator(dev_data)

    base_augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=args.context_window_samples_before,
            windowlen=args.context_window_length,
            selection="random",
            strategy="variable"
        ),
        sbg.RandomWindow(
            windowlen=args.model_input_samples,
            strategy="pad",
            low=0.0 
        ),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, model_labels=model.labels, sigma=args.sigma, dim=0),
        # sbg.ChangeDtype(np.float32) # Keep this here, it should convert 'X' and 'y' if they are numpy arrays.
                                    # The issue might be subtle, or we need explicit casting later.
                                    # Forcing float32 for 'y' will be done explicitly before loss.
    ]
    
    # Apply ChangeDtype at the end of all other processing for X and y
    # This assumes X and y are the primary keys containing numpy arrays to be converted.
    # If ProbabilisticLabeller outputs something other than a numpy array for 'y' that ChangeDtype misses,
    # explicit casting is better. Given the error, explicit casting for y_batch is safest.
    final_dtype_conversion = sbg.ChangeDtype(np.float32)


    # Optional augmentations (applied before final processing steps like Normalize or ProbabilisticLabeller if they affect values)
    # Order matters here. Let's put them before Normalize for now.
    # The very last step before a sample is returned by generator should ideally be type conversion if needed.
    
    # Revised augmentation order
    aug_list_for_gen = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=args.context_window_samples_before,
            windowlen=args.context_window_length,
            selection="random",
            strategy="variable"
        ),
        sbg.RandomWindow(
            windowlen=args.model_input_samples,
            strategy="pad",
            low=0.0 
        )
    ]

    # Optional augmentations for the waveform data ('X')
    if args.add_gain:
        aug_list_for_gen.append(sbg.Gain(min_gain=0.5, max_gain=1.5, probability=0.5, keys=["X"]))
    if args.add_signal_shift:
         aug_list_for_gen.append(sbg.SignalShift(max_shift_samples=args.signal_shift_max, probability=0.5, keys=["X"]))
    if args.add_gaussian_noise:
        aug_list_for_gen.append(sbg.GaussianNoise(std=args.gaussian_noise_std, probability=0.5, keys=["X"]))
    
    # Final processing steps
    aug_list_for_gen.extend([
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, model_labels=model.labels, sigma=args.sigma, dim=0), # Creates 'y'
        sbg.ChangeDtype(np.float32) # Apply to all numpy arrays in sample dict, including 'X' and 'y'
    ])


    train_generator.add_augmentations(aug_list_for_gen)
    dev_generator.add_augmentations(aug_list_for_gen) # Use same for dev

    train_loader = DataLoader(
        train_generator, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_seeding, pin_memory=True, drop_last=True
    )
    dev_loader = DataLoader(
        dev_generator, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_seeding, pin_memory=True
    )

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # criterion = torch.nn.BCELoss() 
    criterion = original_custom_loss_fn
    best_val_loss = float('inf')
    output_model_path = os.path.join(args.output_dir, args.model_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Will save best model to: {output_model_path}")

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        train_epoch_loss = 0
        for batch_id, batch in enumerate(train_loader):
            X_batch = batch["X"].to(device)
            # ** EXPLICITLY CAST y_batch TO FLOAT32 **
            y_batch = batch["y"].to(device).float() 

            if epoch == 0 and batch_id == 0:
                logging.info(f"First training batch X shape: {X_batch.shape} (dtype: {X_batch.dtype}), y shape: {y_batch.shape} (dtype: {y_batch.dtype})")
                if X_batch.shape[-1] != args.model_input_samples:
                     logging.error(f"Data pipeline delivered X_batch with {X_batch.shape[-1]} samples, but args.model_input_samples is {args.model_input_samples}.")
                     return
                if y_batch.dtype != torch.float32 or X_batch.dtype != torch.float32: # This check should now pass for y_batch
                     logging.warning(f"Expected float32 tensors for loss calculation. Got X: {X_batch.dtype}, y: {y_batch.dtype}")


            pred = model(X_batch)
            loss = criterion(pred, y_batch) # Now y_batch is explicitly float32
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                current_samples_processed = (batch_id + 1) * args.batch_size
                logging.info(f"  Batch {batch_id+1}/{len(train_loader)}: Avg Batch Loss: {loss.item():>7f}  [{current_samples_processed:>6d}/{len(train_data):>6d}]")
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.6f}")

        model.eval()
        val_epoch_loss = 0
        with torch.no_grad():
            for batch_id_val, batch in enumerate(dev_loader):
                X_batch = batch["X"].to(device)
                y_batch = batch["y"].to(device).float() # Explicitly cast for validation too
                if epoch == 0 and batch_id_val == 0: 
                    logging.info(f"First validation batch X shape: {X_batch.shape} (dtype: {X_batch.dtype}), y shape: {y_batch.shape} (dtype: {y_batch.dtype})")

                pred = model(X_batch)
                val_loss = criterion(pred, y_batch)
                val_epoch_loss += val_loss.item()
        
        avg_val_loss = val_epoch_loss / len(dev_loader)
        logging.info(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logging.info(f"New best validation loss: {best_val_loss:.6f}. Saving model to {output_model_path}")
            torch.save(model.state_dict(), output_model_path)
        
        if args.save_latest_model_epoch:
            base_name, ext = os.path.splitext(args.model_filename)
            latest_model_filename = f"{base_name}_epoch{epoch+1}_latest{ext if ext else '.pth'}"
            latest_model_path = os.path.join(args.output_dir, latest_model_filename)
            logging.info(f"Saving model from epoch {epoch+1} to {latest_model_path}")
            torch.save(model.state_dict(), latest_model_path)

    logging.info(f"Training complete. Best model (val_loss: {best_val_loss:.6f}) saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PhaseNet for seismic phase picking using SeisBench.")
    
    parser.add_argument("--dataset_name", type=str, default="Iquique", help="Name of the SeisBench dataset.")
    parser.add_argument("--force_download_data", action="store_true", help="Force download dataset.")
    parser.add_argument("--context_window_samples_before", type=int, default=3000, help="Samples before pick for WindowAroundSample.")
    parser.add_argument("--context_window_length", type=int, default=6000, help="Total length of WindowAroundSample context window.")
    parser.add_argument("--model_input_samples", type=int, default=3001, help="The exact number of samples for model input. Standard PhaseNet uses 3001.")

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--sigma", type=float, default=20, help="Sigma for ProbabilisticLabeller.")
    
    parser.add_argument("--use_pretrained_weights", action="store_true", help="Use pre-trained PhaseNet weights.")
    parser.add_argument("--pretrained_model_name", type=str, default="stead", choices=["stead", "instance", "ethz", "geofon"], help="Pre-trained model name.")

    parser.add_argument("--add_gaussian_noise", action="store_true", help="Add Gaussian noise augmentation.")
    parser.add_argument("--gaussian_noise_std", type=float, default=0.05, help="Std dev for Gaussian noise.")
    parser.add_argument("--add_signal_shift", action="store_true", help="Add signal shift augmentation.")
    parser.add_argument("--signal_shift_max", type=int, default=200, help="Max samples for signal shift.")
    parser.add_argument("--add_gain", action="store_true", help="Add gain augmentation.")
    
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--log_interval", type=int, default=20, help="Log training loss every N batches.")

    parser.add_argument("--output_dir", type=str, default="trained_models_phasenet", help="Directory to save models.")
    parser.add_argument("--model_filename", type=str, default="phasenet_best_model.pth", help="Filename for best model.")
    parser.add_argument("--save_latest_model_epoch", action="store_true", help="Save model from each epoch.")

    args = parser.parse_args()
    main(args)