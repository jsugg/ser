import multiprocessing as mp

# Emotions supported by the RAVDESS dataset
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

TMP_FOLDER = "./ser/tmp"

# Model configuration settings
MODELS_CONFIG = {
    'models_folder': './ser/models',
    'whisper_model': {
        'name': 'large-v2',
        'path': 'OpenAI/whisper/'
    },
    'num_cores': mp.cpu_count()
}

DATASET = {
    'folder': './ser/dataset/ravdess',
    'subfolder_prefix': 'Actor_*',
    'extension': '*.wav'
}

TIMELINE_CONFIG = {
    'folder': './ser/transcripts'
}

# Default feature extraction configuration
DEFAULT_FEATURE_CONFIG = {
    'mfcc': True,
    'chroma': True,
    'mel': True,
    'contrast': True,
    'tonnetz': True,
}

# Default application configuration
DEFAULT_CONFIG = {
    'language': 'en',
    'file': None,
    'train': False 
}

# Neural network parameters for MLP Classifier
NN_PARAMS = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08,
    'hidden_layer_sizes': (300,),
    'learning_rate': 'adaptive',
    'max_iter': 500
}

# Audio file read parameters
AUDIO_READ_CONFIG = {
    'max_retries': 3,
    'retry_delay': 1
}

# Extended feature extraction parameters
EXTENDED_FEATURE_PARAMS = {
    'frame_size': 3,  # in seconds
    'frame_stride': 1  # in seconds
}
