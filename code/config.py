#########
# NAMES #
#########

PAD = 'PAD'
EOC = 'EOC'
UNK = 'UNK'
SOS = 'SOS'

CASE_ID = 'case:concept:name'
ACTIVITY = 'concept:name'
RESOURCE = 'org:resource'
TIMESTAMP = 'time:timestamp'


##############
# PARAMETERS #
##############
BATCH_SIZE = 128
T_LEARNING_RATE = [3e-4, 1e-5]#1e-4#1e-3
DEFAULT_EPOCHS = 500
DEFAULT_PATIENCE = 40
STOPPING_METHODS = ['loss', 'val']
DEFAULT_STOPPING_METHOD = 'val'
DEFAULT_EMB_SIZE = 50
DEFAULT_ENCODER_BLOCKS = 4
DEFAULT_DECODER_BLOCKS = 4
DEFAULT_ATTENTION_HEADS = 2



#########
# PATHS #
#########

ROOT_DATA_PATH = 'data/'

############
# DEFAULTS #
############
# - Event Log Defaults

DEFAULT_EVENT_LOG = 'env_permit'
DEFAULT_FOLD = "0"
DEFAULT_FILE_TYPE = ".xes.gz"

# - Model Defaults -
WINDOW_SIZE_METHODS = ['max', 'auto']
DEFAULT_WINDOW_SIZE_METHOD = 'auto'



########
# SEED #
########
SEED = 42

import random
import numpy as np
import torch

def set_seed(seed = SEED):
    """
    Set the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
