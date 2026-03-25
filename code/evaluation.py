from sys import prefix

from data_processors.event_logs import EventLog
from encoders_and_decoders import StepEventTransformer, EventTransformer
import config
import numpy as np
import torch
from utils import get_windows_from_traces, update_traces_with_predictions
import numpy as np
from typing import Union

def damerau_levenshtein_similarity(predictions: np.array, ground_truths: np.array,
                                     code_end: Union[str, int]):
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(ground_truths, np.ndarray):
        ground_truths = np.array(ground_truths)

    if code_end != None:
        try:
            l1 = np.where(predictions == code_end)[0][0].item()
        except (ValueError, IndexError):
            l1 = predictions.size
        try:
            l2 = np.where(ground_truths == code_end)[0][0].item()
        except (ValueError, IndexError):
            l2 = ground_truths.size
    else:
        l1 = predictions.size
        l2 = ground_truths.size

    if max(l1, l2) == 0:
        return 1.0

    matrix = [list(range(l1 + 1))] * (l2 + 1)

    for i in list(range(l2 + 1)):
        matrix[i] = list(range(i, i + l1 + 1))

    for i in range(1, l2 + 1):
        for j in range(1, l1 + 1):
            cost = 0 if predictions[j - 1] == ground_truths[i - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,         # Deletion
                               matrix[i][j - 1] + 1,         # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution

            # Check for transposition
            if i > 1 and j > 1 and predictions[j - 1] == ground_truths[i - 2] and \
                    predictions[j - 2] == ground_truths[i - 1]:
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + cost)  # Transposition

    distance = float(matrix[l2][l1])

    similarity = 1.0 - distance / max(l1, l2)
    return similarity



def test(
    model : StepEventTransformer | EventTransformer, 
    event_log : EventLog, 
    window_size : int, 
    validation : bool = False,
    batch_size : int = config.BATCH_SIZE * 4
    ):
    #self.model.model, self.event_log, self.window_size, evaluation_mode=True, deterministic = True, return_all_evaluations = True
    if isinstance(model, EventTransformer):
        return test_ed(model, event_log, window_size, validation)

    train_model = model.training
    model.eval()
    data = event_log.val_data if validation else event_log.test_data
    cat_prefixes, num_prefixes, targets, prefix_lengths = data.X, data.X_num, data.y, data.lengths
    device = model.device

    similarities = {attribute.name : [] for attribute in event_log.attributes}
    eocs = [attribute.val_to_emb(config.EOC) for attribute in event_log.attributes]
    eoc = eocs[0]

    for index in range(0, len(cat_prefixes), batch_size):
        
        prefix = cat_prefixes[index:index + batch_size]
        batch_size = prefix.shape[0]

        pad_suffix = np.array(event_log.pads)[np.newaxis, :, np.newaxis].repeat(batch_size, axis=0).repeat(event_log.max_trace_length + 1, axis=2)
        pad_suffix_num = np.zeros((batch_size, len(event_log.attributes_num), event_log.max_trace_length))

        traces = torch.tensor(np.concatenate((prefix, pad_suffix), axis=2)).to(device)

        if event_log.numerical_times:
            prefix_num = num_prefixes[index:index + batch_size]
            num_traces = torch.tensor(np.concatenate((prefix_num, pad_suffix_num), axis=2)).to(device)

        trace_lengths = torch.tensor(prefix_lengths[index:index + batch_size], dtype=torch.int64).to(device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_predictions = torch.tensor([event_log.max_trace_length - pl for pl in trace_lengths], 
                                          dtype=torch.int32, device=device)
        suffix_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)

        for _ in range(event_log.max_trace_length):
            if finished.all():
                break

            windows = get_windows_from_traces(traces, trace_lengths, window_size)
            if event_log.numerical_times:
                windows_num = get_windows_from_traces(num_traces, trace_lengths, window_size)
            else:
                windows_num = None

            next_events = model.predict(windows, windows_num)
            traces = update_traces_with_predictions(traces, next_events, trace_lengths)

            trace_lengths += 1 # Count the number of predicted events

            suffix_lengths += ~finished # Count the real suffix predictions
            finished |= (next_events[:, 0] == eoc) | (suffix_lengths >= max_predictions)


        for i in range(batch_size):
            start = prefix_lengths[index + i].item()
            length = suffix_lengths[i].item()
            end = start + length
            suffix = traces[i, :, start:end].cpu().numpy()
            target = targets[index + i]

            for attr_index, attr in enumerate(event_log.attributes):
                similarities[attr.name].append(damerau_levenshtein_similarity(suffix[attr_index], target[attr_index], eocs[attr_index]))
        
    # Compute average similarities
    avg_similarities = {attr: np.mean(sims) for attr, sims in similarities.items()}

    if train_model:
        model.train()

    return avg_similarities


def test_ed(
    model : EventTransformer, 
    event_log : EventLog, 
    window_size : int, 
    validation : bool = False, 
    batch_size : int = config.BATCH_SIZE * 4
    ):
    train_model = model.training
    model.eval()
    data = event_log.val_data if validation else event_log.test_data
    cat_prefixes, num_prefixes, targets, prefix_lengths = data.X, data.X_num, data.y, data.lengths
    device = model.device

    initial_suffix = [attribute.val_to_emb(config.SOS) for attribute in event_log.attributes]
    eocs = [attribute.val_to_emb(config.EOC) for attribute in event_log.attributes]
    eoc = eocs[0]

    similarities = {attribute.name : [] for attribute in event_log.attributes}

    for index in range(0, len(cat_prefixes), batch_size):
        
        prefix = cat_prefixes[index:index + batch_size]
        numerical_prefix = None
        target = targets[index:index + batch_size, :, 1:] # Remove SOS from target
        current_prefix_lengths = prefix_lengths[index:index + batch_size]
        batch_size = prefix.shape[0]

        windows = torch.tensor(prefix, dtype=torch.int64).to(device)
        windows = get_windows_from_traces(windows, torch.tensor(current_prefix_lengths).to(device), window_size)
        num_windows = None
        suffixes = torch.tensor(initial_suffix, dtype=torch.int64).repeat(batch_size, 1).unsqueeze(2).to(device)
        suffixes_num = None

        encoder_output, encoder_mask = model.encode(windows, num_windows)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_predictions = torch.tensor([event_log.max_trace_length - pl for pl in current_prefix_lengths], #todo +1?
                                        dtype=torch.int32, device=device)
        current_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)

        for _ in range(event_log.max_trace_length):
            if finished.all():
                break
            
            next_event = model.predict(encoder_output, encoder_mask, suffixes, suffixes_num)

            suffixes = torch.cat((suffixes, next_event.unsqueeze(2)), dim=2)

            current_lengths += ~finished
            finished |= (next_event[:, 0] == eoc) | (current_lengths >= max_predictions)

        suffixes = suffixes[:, :, 1:]

        for i in range(batch_size):
            eoc_positions = (suffixes[i, 0, :] == eoc).nonzero(as_tuple=True)[0]
            if len(eoc_positions) > 0:
                end_idx = min(eoc_positions[0].item() + 1, max_predictions[i].item())
            else:
                end_idx = max_predictions[i].item()

            for attr_idx, attribute in enumerate(event_log.attributes):
                predicted_suffix = suffixes[i, attr_idx, :end_idx].cpu().numpy()

                similarity = damerau_levenshtein_similarity(target[i,attr_idx], predicted_suffix, eocs[attr_idx])
                if attribute.name not in similarities:
                    similarities[attribute.name] = []
                similarities[attribute.name].append(similarity)
        
    # Compute average similarities
    avg_similarities = {attr: np.mean(sims) for attr, sims in similarities.items()}

    if train_model:
        model.train()

    return avg_similarities
