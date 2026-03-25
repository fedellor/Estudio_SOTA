import config
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def create_embedding_dict(pad, data, eoc, special=None):
    unique_values = sorted(data.astype(str).unique())
    attribute_dict = {}
    # 0 is reserved for padding
    attribute_dict[pad] = 0
    for i in range(len(unique_values)):
        attribute_dict[unique_values[i]] = i + 1
    attribute_dict[eoc] = len(unique_values) + 1
    if special:
        if type(special) == list:
            for s in range(len(special)):
                attribute_dict[special[s]] = len(unique_values) + 2 + s
        elif type(special) == str:
            attribute_dict[special] = len(unique_values) + 2
    return attribute_dict, len(attribute_dict.keys())


class Attribute():
    CATEGORICAL = 0
    NUMERIC = 1
    """
    By default all attributes are categorical.
    If you want to create a numeric attribute, you need to pass a scaler.
    """
    def __init__(self, name, data=None, pad=config.PAD, eoc=config.EOC, special=None, verbose=False, is_numerical=False):

        self.name = name

        if is_numerical:
            self.type = Attribute.NUMERIC
            self.low = float(np.min(data))
            scaler = StandardScaler()
            scaler.fit(pd.DataFrame(data))
            self.mean = float(scaler.mean_[0])
            self.scale = float(scaler.scale_[0])
        else:
            self.type = Attribute.CATEGORICAL
            self.dict, self.dict_size = create_embedding_dict(pad, data, eoc, special)
            self.low = 0
            self.high = self.dict_size
            if verbose:
                print(f"[INFO] {self.name} attribute has {self.dict_size} unique values")

    NUMERIC_SPECIALS = [config.PAD, config.SOS, config.EOC, config.UNK]
    def val_to_emb(self, data):
        if self.type == Attribute.NUMERIC:
            if data in Attribute.NUMERIC_SPECIALS:
                return 0.0
            
            return max(min((data - self.mean) / self.scale, 3.0), -3.0)
            #return (data - self.mean) / self.scale
            
        
        return self.dict[str(data)]
    

    def to_dict(self):
        if self.type == Attribute.NUMERIC:
            return {
                "name": self.name,
                "pad": self.val_to_emb(config.PAD),
                "eoc": self.val_to_emb(config.EOC),
                "type": self.type,
                "low": self.low,
            }
        return {
            "name": self.name,
            "dict_size": self.dict_size,
            "pad": self.val_to_emb(config.PAD),
            "eoc": self.val_to_emb(config.EOC),
            "type": self.type,
            "low": self.low,
            
        }