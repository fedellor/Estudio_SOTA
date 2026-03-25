import torch
import torch.nn as nn
from encoders_and_decoders import StepEventTransformer, EventTransformer
from data_processors.event_logs import EventLog
import config
import os
from utils import get_windows_from_traces
from tqdm import tqdm
from evaluation import test


def calculate_loss(
    model : StepEventTransformer | EventTransformer,
    event_log : EventLog,
    dataloader : torch.utils.data.DataLoader = None, 
    optimizer : torch.optim.Optimizer = None,
    clip_loss : bool = False,
    validation : bool = False,
    ):

    pad = event_log.attributes[0].val_to_emb(config.PAD)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad)

    if validation:
        model.eval()
    else:
        model.train()

    epoch_loss = 0.0
    epoch_batches = 0
    if isinstance(model, StepEventTransformer):
        for x_cat, x_num, y in dataloader:
            if optimizer:
                optimizer.zero_grad()
            outputs = model(x_cat, x_num)

            loss = 0.0

            for i, logits in enumerate(outputs):
                flat_logits = logits.reshape(-1, event_log.vocab_sizes[i])
                flat_target = y[:, i, :].reshape(-1)
                loss = loss + loss_fn(flat_logits, flat_target)

            epoch_loss += loss.item()
            epoch_batches += 1
            if not validation:
                loss.backward()
                if clip_loss:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
    else:
        for x_cat, x_num, y in dataloader:
            if optimizer:
                optimizer.zero_grad()
            outputs = model(x_cat, x_num, y[:, :, :-1]) #? Teacher forcing: provide the target sequence except the last token

            loss = 0.0
            for i, logits in enumerate(outputs):
                flat_logits = logits.reshape(-1, event_log.vocab_sizes[i])
                flat_target = y[:, i, 1:].reshape(-1)
                loss = loss + loss_fn(flat_logits, flat_target)

            epoch_loss += loss.item()
            epoch_batches += 1
            if not validation:
                loss.backward()
                if clip_loss:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            

    epoch_loss /= epoch_batches

    return epoch_loss


def fit(
    model : StepEventTransformer | EventTransformer, 
    event_log : EventLog, 
    window_size : int, 
    batch_size : int = config.BATCH_SIZE,
    epochs : int = 200,
    stopping_patience : int = 40,
    store_path : str = f"{config.ROOT_DATA_PATH}/models/",
    force_new_model : bool = False,
    lr = config.T_LEARNING_RATE,
    clip_loss = True,
    stop_with_loss = True,
    ):
    
    device = model.device
    model_path = f"{store_path}/{event_log.name}_fold{event_log.fold}{'step' if isinstance(model, StepEventTransformer) else ''}_event_predictor.pth"
    os.makedirs(store_path, exist_ok=True)

    if os.path.exists(model_path) and not force_new_model:
        try:
            print(f"Loading existing model from {model_path}")
            model.load_state_dict(torch.load(model_path, weights_only=True))
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}. Training a new model.")

    train_dataloader = generate_dataloader(event_log, device, window_size, batch_size, validation=False)
    val_dataloader = generate_dataloader(event_log, device, window_size, batch_size, validation=True) if stop_with_loss else None
    optimizer, scheduler = create_optimizer(lr, model, epochs)

    epochs_witout_improvement = 0
    best_metric = float('inf') if stop_with_loss else stop_with_loss

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        epoch_loss = calculate_loss(model, event_log, train_dataloader, optimizer=optimizer, clip_loss=clip_loss, validation=False)

        #Validation and early stopping
        if stop_with_loss:
            with torch.no_grad():
                val_loss = calculate_loss(model, event_log, val_dataloader, optimizer=optimizer, clip_loss=False, validation=True)
            if val_loss < best_metric:
                best_metric = val_loss
                epochs_witout_improvement = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_witout_improvement += 1
                if epochs_witout_improvement >= stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            info = {'val_loss': f'{val_loss:.4f}', 'best_val': f'{best_metric:.4f}'}

        else:
            info = test(model, event_log, window_size, validation=True)
            activity_similarities = info[config.ACTIVITY]
            if activity_similarities > best_metric:
                best_metric = activity_similarities
                epochs_witout_improvement = 0
                torch.save(model.state_dict(), model_path)
            else:
                epochs_witout_improvement += 1
                if epochs_witout_improvement >= stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            info['best_val'] = f'{best_metric:.4f}'


        info['early_stopping'] = f'{epochs_witout_improvement}/{stopping_patience}'
        info['train_loss'] = f'{epoch_loss:.4f}'
        info['lr'] = optimizer.param_groups[0]['lr'] if scheduler is not None else lr
        pbar.set_description(f"Epoch {epoch+1}/{epochs}")
        pbar.set_postfix(info)

        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    return model



def create_optimizer(
    lr : float | list, 
    model : StepEventTransformer | EventTransformer, 
    epochs : int
    ):
    if isinstance(lr, list):
        if len(lr) != 2:
            raise ValueError("Learning rate list must have two elements: [initial_lr, final_lr]")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr[0],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr[1]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    
    return optimizer, scheduler



def generate_dataloader(
    event_log : EventLog,
    device : str,
    window_size : int,
    batch_size : int = config.BATCH_SIZE,
    validation : bool = False
    ):

    data = event_log.val_data if validation else event_log.train_data

    windows = get_windows_from_traces(torch.tensor(data.X).to(device), torch.tensor(data.lengths).to(device), window_size)
    if event_log.numerical_times:
        windows_num = get_windows_from_traces(torch.tensor(data.X_num).to(device), torch.tensor(data.lengths).to(device), window_size)
    else:
        windows_num = torch.empty(data.X.shape[0]) #todo poner None
    
    targets = torch.tensor(data.y).to(device)
    dataset = torch.utils.data.TensorDataset(windows, windows_num, targets) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader