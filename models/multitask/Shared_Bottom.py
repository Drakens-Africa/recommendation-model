import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input

def Shared_Bottom(dnn_feature_columns, num_tasks=None, task_types=None, task_names=None, 
    bottom_dnn_units=[128, 128], tower_dnn_units_lists=[[64, 32], [64, 32]], 
    l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    