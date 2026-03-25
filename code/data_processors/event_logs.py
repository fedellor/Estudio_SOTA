import config
import pm4py
import pandas as pd
import yaml
from data_processors.attributes import Attribute
from tqdm import tqdm
import numpy as np
import json


def read_csv(file_path, return_legacy_log_object = False):
    data = pd.read_csv(file_path)

    column_mapping = {
        'case_id': config.CASE_ID,
        'activity': config.ACTIVITY,
        'Start_Time': config.TIMESTAMP
    }

    if 'resource' in data.columns:
        column_mapping['resource'] = config.RESOURCE

    data = data.rename(columns=column_mapping)
    data['time:timestamp'] = pd.to_datetime(data['time:timestamp'], format='mixed')
    data[config.CASE_ID] = data[config.CASE_ID].astype(str)

    if return_legacy_log_object:
        event_log = pm4py.convert_to_event_log(data)
        return event_log
    
    return data



class BaseEventLog:
    ROOT_DATA_PATH = f"{config.ROOT_DATA_PATH}event_log/"
    SPLITS_DATA_PATH = ROOT_DATA_PATH + "splitted/"

    VALID_TIME_ATTRIBUTES = ['dow', 'tod', 'wd']  # day of week, time of day, working day

    def __init__(self, 
                 name : str = config.DEFAULT_EVENT_LOG, 
                 fold : str = config.DEFAULT_FOLD,
                 file_type : str = config.DEFAULT_FILE_TYPE,
                 attributes : list | None = None,
                 time_attributes : list = [],
                 numerical_times : bool = False,
                 start_of_suffix : bool = False,
                 verbose : bool = False):
        
        """
        name : str
            Name of the event log (without file extension).
        fold : str
            Fold number for cross-validation (e.g., '0', '1', '2', '3', '4').
        file_type : str
            File extension of the event log (e.g., '.xes.gz', '.csv').
        attributes : list or str
            List of attribute names to consider or a comma-separated. None for all attributes.
        time_attributes : List
            List of time attributes to be calculated: 'dow', 'tod', 'wd'
        numerical_times : bool
            Whether to calculate numerical time differences.
        start_of_suffix : bool
            Whether to include a start of trace event.
        """
        
        self.name = name
        self.fold = fold
        self.file_type = file_type
        self.use_sos = start_of_suffix
        self.numerical_times = numerical_times

        self.time_attributes = [attr for attr in time_attributes if attr in BaseEventLog.VALID_TIME_ATTRIBUTES]

        if attributes is None:
            self.selected_attributes = None
        else:
            self.selected_attributes = attributes.split(",") if isinstance(attributes, str) else attributes

            if config.ACTIVITY not in self.selected_attributes:
                self.selected_attributes = [config.ACTIVITY] + self.selected_attributes

        old_setting = pm4py.utils.constants.SHOW_PROGRESS_BAR
        pm4py.utils.constants.SHOW_PROGRESS_BAR = verbose
        self.__load_data__()
        pm4py.utils.constants.SHOW_PROGRESS_BAR = old_setting

        self.__preprocess_data__(time_attributes, verbose)

        # List of variables to be used
        # - self.name, self.fold
        # - self.pads, self.vocab_sizes
        # - self.attributes, self.attributes_num
        # - self.train_data, self.val_data, self.test_data


    def __calculate_numerical_times_to__(self, data):
        for trace in data:
            start_time = trace[0][config.TIMESTAMP]
            prev_time = start_time
            for event in trace:
                timestamp = event[config.TIMESTAMP]
                event['time_from_start'] = (timestamp - start_time).total_seconds()
                event['time_from_prev'] = (timestamp - prev_time).total_seconds()
                prev_time = timestamp


    def __calculate_numerical_times__(self):
        self.__calculate_numerical_times_to__(self.train_log)
        self.__calculate_numerical_times_to__(self.val_log)
        self.__calculate_numerical_times_to__(self.test_log)


    def __calculate_time_attributes__(self, time_attributes):
        if 'dow' in time_attributes:
            self.df['dow'] = self.df[config.TIMESTAMP].dt.dayofweek.astype(str)

            def calculate_dow(log):
                for trace in log:
                    for event in trace:
                        timestamp = event[config.TIMESTAMP]
                        event['dow'] = str(timestamp.weekday())
                return log
            
            self.train_log = calculate_dow(self.train_log)
            self.val_log = calculate_dow(self.val_log)
            self.test_log = calculate_dow(self.test_log)


        if 'tod' in time_attributes:
            def get_tod(hour):
                if 0 <= hour < 8:
                    return '0'
                elif 8 <= hour < 12:
                    return '1'
                elif 12 <= hour < 16:
                    return '2'
                elif 16 <= hour < 20:
                    return '3'
                else:
                    return '4'
                
            self.df['tod'] = self.df[config.TIMESTAMP].dt.hour.apply(get_tod).astype(str)
            def calculate_tod(log):
                for trace in log:
                    for event in trace:
                        timestamp = event[config.TIMESTAMP]
                        hour = timestamp.hour
                        event['tod'] = get_tod(hour)
                return log
            self.train_log = calculate_tod(self.train_log)
            self.val_log = calculate_tod(self.val_log)
            self.test_log = calculate_tod(self.test_log)

        if 'wd' in time_attributes:
            self.df['wd'] = (~self.df[config.TIMESTAMP].dt.dayofweek.isin([5,6])).astype(int).astype(str)

            def calculate_wd(log):
                for trace in log:
                    for event in trace:
                        timestamp = event[config.TIMESTAMP]
                        event['wd'] = '1' if timestamp.weekday() < 5 else '0'
                return log
            
            self.train_log = calculate_wd(self.train_log)
            self.val_log = calculate_wd(self.val_log)
            self.test_log = calculate_wd(self.test_log)


    def __preprocess_data__(self, time_attributes, verbose):
        
        if self.numerical_times:
            self.__calculate_numerical_times__()
        self.__calculate_time_attributes__(time_attributes)

        self.attributes = []
        self.vocab_sizes, self.pads = [], []
        
        for attribute_name in self.attribute_names + self.time_attributes:
            #attribute = Attribute(attribute_name, self.df[attribute_name], special=config.UNK if attribute_name != config.ACTIVITY else None)
            attribute = Attribute(attribute_name, self.df[attribute_name], special=config.SOS if self.use_sos else None, verbose=verbose)
            self.attributes.append(attribute)

            self.vocab_sizes.append(attribute.dict_size)
            self.pads.append(attribute.val_to_emb(config.PAD))

        self.attributes_num = []
        for attribute_name in self.attribute_names_num:
            data = []
            for trace in self.train_log:
                for event in trace:
                    data.append(event[attribute_name])
            #needs to use train data to fit scaler
            attribute = Attribute(attribute_name, data, is_numerical=True, verbose=verbose)
            self.attributes_num.append(attribute)
        
        self.__load_samples__(verbose)


    def __load_samples__(self):
        raise NotImplementedError("Subclasses should implement this!")

    
    def __load_data__(self):
        
        read_file = pm4py.read_xes if self.file_type == ".xes.gz" else read_csv

        self.df = read_file(BaseEventLog.ROOT_DATA_PATH+self.name+self.file_type, return_legacy_log_object = False) # Dataframe

        self.train_log = read_file(BaseEventLog.SPLITS_DATA_PATH + 'train_fold' + self.fold + "_" + self.name + self.file_type, return_legacy_log_object = True) #EventLog
        self.val_log = read_file(BaseEventLog.SPLITS_DATA_PATH + 'val_fold' + self.fold + "_" + self.name + self.file_type, return_legacy_log_object = True) #EventLog
        self.test_log = read_file(BaseEventLog.SPLITS_DATA_PATH + 'test_fold' + self.fold + "_" + self.name + self.file_type, return_legacy_log_object = True) #EventLog

        with open(BaseEventLog.ROOT_DATA_PATH+"/attributes.yaml", "r") as file:
            attribute_yaml = yaml.safe_load(file)

        self.attribute_names = [config.ACTIVITY]

        if self.name in attribute_yaml:
            if self.selected_attributes is None:
                self.attribute_names += attribute_yaml[self.name]
            else:
                for attribute in attribute_yaml[self.name]:
                    if attribute in self.selected_attributes and attribute not in self.attribute_names:
                        self.attribute_names.append(attribute)

        self.attribute_names_num = ['time_from_start', 'time_from_prev'] if self.numerical_times else []

        self.df = self.df[[config.CASE_ID] + self.attribute_names + [config.TIMESTAMP]]

        group_by_id_size = self.df.groupby(config.CASE_ID).size()
        self.max_trace_length = group_by_id_size.max()
        self.mean_trace_length = group_by_id_size.mean()
        self.std_trace_length = group_by_id_size.std()
        self.min_trace_length = group_by_id_size.min()

        



    def to_dict(self):
        return {
            "cat_attributes" : [attribute.to_dict() for attribute in self.attributes],
            "num_attributes" : [attribute.to_dict() for attribute in self.attributes_num],
            "max_trace_length" : self.max_trace_length,
            "mean_trace_length" : self.mean_trace_length,
            "eoc" : [attribute.val_to_emb(config.EOC) for attribute in self.attributes],
            "vocab_sizes" : self.vocab_sizes,
            "pads" : self.pads,
            "name": self.name,
            "fold": self.fold,
            #"preceded_activities": self.get_preceded_activities()
        }


class EventLogData:
    def __init__(self, X, X_num, y, y_num, lengths):
        self.X = X
        self.X_num = X_num
        self.y = y
        self.y_num = y_num
        
        self.lengths = lengths



class EventLog(BaseEventLog):
    ROOT_DATA_PATH = f"{config.ROOT_DATA_PATH}event_log/"
    SPLITS_DATA_PATH = ROOT_DATA_PATH + "splitted/"

    def __init__(self, 
                 name = config.DEFAULT_EVENT_LOG,
                 fold = config.DEFAULT_FOLD,
                 file_type = config.DEFAULT_FILE_TYPE,
                 attributes = None,
                 time_attributes = [],
                 numerical_times = False,
                 start_of_suffix = False,
                 verbose = False):
        
        """
        name : str
            Name of the event log (without file extension).
        fold : str
            Fold number for cross-validation (e.g., '0', '1', '2', '3', '4').
        file_type : str
            File extension of the event log (e.g., '.xes.gz', '.csv').
        attributes : list or str
            List of attribute names to consider or a comma-separated. None for all attributes.
        time_attributes : List
            List of time attributes to be calculated: 'dow', 'tod', 'wd'
        numerical_times : bool
            Whether to calculate numerical time differences.
        start_of_suffix : bool
            Whether to include a start of trace event.
        """
        super().__init__(name, fold, file_type, attributes, time_attributes, numerical_times, start_of_suffix, verbose)


    def extract_samples(self, event_log, verbose = False):
        prefixes = []
        suffixes = []
        prefixes_num = []
        suffixes_num = []
        prefixes_lengths = []

        pbar = tqdm(len(event_log), desc="Extracting samples") if verbose else None

        for trace in event_log:
            for prefix_index in range(len(trace)):
                prefixes_lengths.append(prefix_index + 1)

                current_prefix = []
                current_suffix = []
                current_prefix_num = []
                current_suffix_num = []

                for attribute in self.attributes + self.attributes_num:
                    attribute_name = attribute.name

                    attribute_prefix = [attribute.val_to_emb(event[attribute_name]) for event in trace[:prefix_index+1]] +\
                                        [attribute.val_to_emb(config.PAD)] * (self.max_trace_length - prefix_index - 1)

                    attribute_prefix = attribute_prefix[-self.max_trace_length:]

                    attribute_suffix = [attribute.val_to_emb(config.SOS)] if self.use_sos else []
                    attribute_suffix +=  [attribute.val_to_emb(event[attribute_name]) for event in trace[prefix_index+1:]] +\
                                         [attribute.val_to_emb(config.EOC)]
                    attribute_suffix += [attribute.val_to_emb(config.PAD)] * (self.max_trace_length - len(attribute_suffix) + (1 if self.use_sos else 0)) # +1 for SOS

                    if attribute.type == Attribute.CATEGORICAL:
                        current_prefix.append(attribute_prefix)
                        current_suffix.append(attribute_suffix)
                    else:
                        current_prefix_num.append(attribute_prefix)
                        current_suffix_num.append(attribute_suffix)

                
                prefixes.append(current_prefix)
                suffixes.append(current_suffix)
                prefixes_num.append(current_prefix_num)
                suffixes_num.append(current_suffix_num)

            if verbose:
                pbar.update(1)
        
        prefixes = np.array(prefixes)
        suffixes = np.array(suffixes)
        prefixes_num = np.array(prefixes_num)
        suffixes_num = np.array(suffixes_num)
        prefixes_lengths = np.array(prefixes_lengths)

        return EventLogData(prefixes, prefixes_num, suffixes, suffixes_num, prefixes_lengths)
    


    def __load_samples__(self, verbose = False):
        self.train_data = self.extract_samples(self.train_log, verbose)
        self.val_data = self.extract_samples(self.val_log, verbose)
        self.test_data = self.extract_samples(self.test_log, verbose)



def get_window_size(window_size, event_log):
    if window_size == "max":
        return event_log.max_trace_length
    elif window_size == "auto":
        #Se lee el archivo json: BaseEventLog.ROOT_DATA_PATH+"/window_sizes.json"
        with open(BaseEventLog.ROOT_DATA_PATH+"/window_sizes.json", "r") as file:
            window_sizes = json.load(file)

        if event_log.name in window_sizes and event_log.fold in window_sizes[event_log.name]:
            return window_sizes[event_log.name][event_log.fold]
        else:
            print(f"[WARNING] Window size for log {event_log.name} fold {event_log.fold} not found in window_sizes.json. Using max trace length: {event_log.max_trace_length}")
            return event_log.max_trace_length
            
    else:
        raise ValueError(f"Unknown window size method: {window_size}")