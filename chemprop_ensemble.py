import os
import json
import torch
import pandas as pd
from rdkit import Chem
from chemprop import data, models, nn
from chemprop.models import save_model
from chemprop.nn.metrics import MAE, MSE
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


from config_v6 import (
    TRAINING_DATA,
    CHALLENGE_TEST_DATA,
    CHALLENGE_ENDPOINTS,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    LOG_DIR,
    EPOCHS,
    MODEL_CONFIGS,
    PATIENCE,
    BATCH_SIZE,
    SMILES_COL,
    TRAINING_TARGETS,
    LOG_NAME,
)

class MultiTaskTrackingMPNN(models.MPNN):
    def __init__(self, task_names, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_names = task_names

    def validation_step(self, batch, batch_idx):
        # 1. Run the standard parent validation step
        loss = super().validation_step(batch, batch_idx)

        # 2. Get predictions and targets
        preds = self(batch.bmg)
        targets = batch.Y
        current_batch_size = targets.size(0)

        # 3. Unscale data
        if self.predictor.output_transform:
            preds = self.predictor.output_transform(preds)
            targets = self.predictor.output_transform(targets)

        # 4. Calculate and log MAE per prediction head
        diff = torch.abs(preds - targets)

        for i, name in enumerate(self.task_names):
            
            mask = ~torch.isnan(targets[:, i])
            
            if mask.sum() > 0:
                # Calculate mean only on valid indices
                task_mae = diff[mask, i].mean()
                
                # Log to TensorBoard
                self.log(
                    f"val/mae_{name}", 
                    task_mae, 
                    on_step=False, 
                    on_epoch=True, 
                    prog_bar=False,
                    batch_size=current_batch_size,
                )
                self.log(f"val/mae_{name}", task_mae, on_step=False, on_epoch=True, prog_bar=False)

        return loss


def load_all_datapoints():
    """Helper function to load dataset for Chemprop"""
    input_df = pd.read_csv(TRAINING_DATA)
    
    # It's safer to use list comprehension for robust error handling if a SMILES is invalid
    smiles = input_df[SMILES_COL].values
    ys = input_df[TRAINING_TARGETS].values
    
    datapoints = []
    for smi, y in zip(smiles, ys):
        mol = Chem.MolFromSmiles(smi)
        if mol: # Basic check
            datapoints.append(data.MoleculeDatapoint(mol, y))
            
    return datapoints


def get_fold_loaders(all_datapoints, train_idx, val_idx, batch_size):
    """Get dataloaders for specific fold"""
    train_data = [all_datapoints[i] for i in train_idx]
    val_data = [all_datapoints[i] for i in val_idx]
    
    train_dset = data.MoleculeDataset(train_data)
    val_dset = data.MoleculeDataset(val_data)
    
    # Scale targets based on TRAIN set only
    output_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(output_scaler)
    
    # Save the transform for this specific fold
    output_transform = nn.transforms.UnscaleTransform.from_standard_scaler(output_scaler)

    train_loader = data.build_dataloader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader = data.build_dataloader(val_dset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, output_transform


def model_ensembling():
    """Train ensemble of Chemprop MPNNs"""
    from lightning.pytorch.loggers import TensorBoardLogger
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    # Load all datapoints
    all_datapoints = load_all_datapoints()
    
    # Create stratified splits
    n_splits = len(MODEL_CONFIGS) 
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, train_size=0.9, random_state=42)
    ys = pd.DataFrame([d.y for d in all_datapoints])
    indicator_matrix = ys.notnull().astype(int).values
    train_test_iterator = msss.split(all_datapoints, indicator_matrix)
    
    ensemble = []
    
    # Create list of MPNN (ensemble members)
    for i, config in enumerate(MODEL_CONFIGS):
        train_idx, val_idx = next(train_test_iterator)
        
        train_loader, val_loader, train_output_transform = get_fold_loaders(
            all_datapoints, 
            train_idx, 
            val_idx, 
            BATCH_SIZE
        )

        mp = nn.BondMessagePassing(
            d_h = config["message_hidden_dim"],
            depth = config["depth"],
            dropout = config["dropout"],
        )
        
        if config['aggregation'] == 'mean':
            aggregation = nn.MeanAggregation()
        elif config['aggregation'] == 'sum':
            aggregation = nn.SumAggregation()
        elif config['aggregation'] == 'norm':
            aggregation = nn.NormAggregation(norm=config['aggregation_norm'])
        elif config['aggregation'] == 'attentive':
            aggregation = nn.AttentiveAggregation(output_size=config['message_hidden_dim'])
        else:
            raise ValueError(f"Unknown aggregation method: {config['aggregation']}")
        
        if config['criterion'] == 'MAE':
            criterion = MAE()
        else:
            criterion = MSE()    
        
        ff = nn.RegressionFFN(
            n_tasks=len(TRAINING_TARGETS),
            input_dim= config["message_hidden_dim"],
            hidden_dim=config["ff_hidden_dim"],
            n_layers=config["ff_layer"],
            dropout=config["dropout"],
            output_transform=train_output_transform,
            criterion=criterion
        )
        
        model = MultiTaskTrackingMPNN(
            task_names=TRAINING_TARGETS,
            message_passing=mp,
            agg=aggregation,
            predictor=ff,
        )
        
        ensemble.append((model, train_loader, val_loader))
        
    ensemble_scores = {}
    # Train each ensemble member
    for i, (model, train_loader, val_loader) in enumerate(ensemble):
        logger = TensorBoardLogger(save_dir=LOG_DIR, name=LOG_NAME, version=f"model_{i}")
        task_checkpoints = []

        callbacks_list = [
                ModelCheckpoint(
                    dirpath=CHECKPOINT_DIR,
                    filename=f'model_{i}_best_avg',
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=False, # Save full model (safer for complex objects)
                ),
                EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=PATIENCE, verbose=False, mode='min')
            ]

        # Create per head checkpoints
        for task_name in TRAINING_TARGETS:
            # Clean task name for filename
            safe_name = task_name.replace("/", "_").replace(" ", "")
            
            task_ckpt = ModelCheckpoint(
                    dirpath=CHECKPOINT_DIR,
                    filename=f'model_{i}_best_{safe_name}',
                    monitor=f"val/mae_{task_name}", 
                    mode="min", 
                    save_top_k=1, # Keep only the absolute best for this task
                    save_weights_only=False
                )
                      
            callbacks_list.append(task_ckpt)
            task_checkpoints.append((task_name, task_ckpt))

        # Train the model
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator='gpu',
            devices=1,
            enable_progress_bar=True,
            logger=logger,
            enable_checkpointing=True,
            callbacks=callbacks_list,
        )
        
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        # Record best val scores
        model_scores = {}
        for task_name, ckpt in task_checkpoints:
            score = ckpt.best_model_score
            if score is not None:
                model_scores[task_name] = float(score)
            else:
                model_scores[task_name] = 1000.0
        
        ensemble_scores[f'model_{i}'] = model_scores
        
        # Save best model heads
        for task_name in TRAINING_TARGETS:
            safe_name = task_name.replace("/", "_").replace(" ", "")
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'model_{i}_best_{safe_name}.ckpt')
            best_model = model.load_from_checkpoint(ckpt_path, weights_only=False)
            save_model(os.path.join(CHECKPOINT_DIR, f'model_{i}_best_{safe_name}_final.pt'), best_model)
        
        # Save transform
        transform_path = os.path.join(CHECKPOINT_DIR, f'transform_{i}.pt')
        torch.save(train_output_transform, transform_path)
        
    # Save ensemble scores to JSON
    scores_path = os.path.join(RESULTS_DIR, "ensemble_model_scores.json")
    with open(scores_path, 'w') as f:
        json.dump(ensemble_scores, f, indent=4)
        
        
def predict_test_set():
    """After training ensemble, load models and predict blind test set"""

    # Datasets
    test_df = pd.read_csv(CHALLENGE_TEST_DATA)
    smiles = test_df[SMILES_COL].values
    test_datapoints = [data.MoleculeDatapoint(Chem.MolFromSmiles(smi)) for smi in smiles]
    test_dset = data.MoleculeDataset(test_datapoints)
    test_loader = data.build_dataloader(test_dset, batch_size=BATCH_SIZE, shuffle=False)

    # Load validation scores
    scores_path = os.path.join(RESULTS_DIR, "ensemble_model_scores.json")
    with open(scores_path, 'r') as f:
        loaded_score = json.load(f)

    final_columns = {}
    
    # Iterate over endpoints
    for task in TRAINING_TARGETS:
        task_preds_across_ensemble = []
        task_weights = []

        # Iterate over ensemble
        for i in range(len(MODEL_CONFIGS)):
            # 1. Load Model
            safe_name = task.replace("/", "_").replace(" ", "")
            pt_path = os.path.join(CHECKPOINT_DIR, f'model_{i}_best_{safe_name}_final.pt')
            
            model = MultiTaskTrackingMPNN.load_from_file(pt_path)
            model.eval()
            
            # 2. Load the corresponding scaler/transform
            transform_path = os.path.join(CHECKPOINT_DIR, f'transform_{i}.pt')
            output_transform = torch.load(transform_path, weights_only=False)
            
            # 3. Predict
            trainer = pl.Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1 if torch.cuda.is_available() else None,
                logger=False, # Disable logging for prediction to reduce clutter
                enable_progress_bar=False
            )
            
            # Get raw (scaled) predictions from the model
            batch_preds = trainer.predict(model, dataloaders=test_loader)
            raw_preds = torch.concat(batch_preds)
            
            # 4. Unscale the predictions
            unscaled_preds = output_transform(raw_preds)

            task_idx = TRAINING_TARGETS.index(task)
            task_col = unscaled_preds[:, task_idx]
            task_preds_across_ensemble.append(task_col)

            # Aggregate weight based on val MAE
            if loaded_score:
                mae = loaded_score[f"model_{i}"].get(task, 1.0)
                weight = 1.0 / (mae + 1e-8)
                task_weights.append(weight)
            else:
                task_weights.append(1.0)

        # All predictions from one endpoint
        stacked_predictions = torch.stack(task_preds_across_ensemble)
        
        # Normalize weights tensor
        w_tensor = torch.tensor(task_weights, device=stacked_predictions.device, dtype=stacked_predictions.dtype)
        w_tensor = w_tensor / w_tensor.sum() # Normalize
        w_tensor = w_tensor.view(-1, 1) # Reshape for broadcasting
        
        avg_task_pred = (stacked_predictions * w_tensor).sum(dim=0)
        
        # Add weighted average of endpoint
        final_columns[task] = avg_task_pred.view(-1)
        print(f"Tensor dimentions for task {task}: {final_columns[task].shape}")

    # Combine all columns into a single tensor
    predictions = torch.stack([final_columns[task] for task in TRAINING_TARGETS], dim=1)

    # transfrom predctions back to original scale from log scale execpt for LogD (first column)
    predictions[:, 1:] = 10 ** predictions[:, 1:]
    
    # # Save predictions
    results_df = pd.DataFrame(predictions.numpy(), columns=TRAINING_TARGETS)
    results_df.drop(columns=[col for col in TRAINING_TARGETS if col != "LogD" and col.replace("log_", "") not in CHALLENGE_ENDPOINTS], inplace=True)
    results_df.insert(0, SMILES_COL, smiles)
    results_df.insert(0, "Molecule Name", test_df["Molecule Name"])
    results_df.columns = ["Molecule Name", SMILES_COL] + CHALLENGE_ENDPOINTS
    
    results_path = os.path.join(RESULTS_DIR, "chemprop_ensemble_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")
    
    # Print mean of each column
    print()
    col_mean = {col: results_df[col].mean() for col in CHALLENGE_ENDPOINTS}
    for c in col_mean:
        print(c, col_mean[c])
    
if __name__ == "__main__":
    # Train ensemble models
    model_ensembling()

    # Predict test set with ensemble
    predict_test_set()