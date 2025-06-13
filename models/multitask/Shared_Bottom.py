import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input

def Shared_Bottom(dnn_feature_columns, num_tasks=None, task_types=None, task_names=None, 
    bottom_dnn_units=[128, 128], tower_dnn_units_lists=[[64, 32], [64, 32]], 
    l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """_summary_

    :param _type_ dnn_feature_columns: _description_
    :param _type_ num_tasks: _description_, defaults to None
    :param _type_ task_types: _description_, defaults to None
    :param _type_ task_names: _description_, defaults to None
    :param list bottom_dnn_units: _description_, defaults to [128, 128]
    :param list tower_dnn_units_lists: _description_, defaults to [[64, 32], [64, 32]]
    :param float l2_reg_embedding: _description_, defaults to 0.00001
    :param int l2_reg_dnn: _description_, defaults to 0
    :param int seed: _description_, defaults to 1024
    :param int dnn_dropout: _description_, defaults to 0
    :param str dnn_activation: _description_, defaults to 'relu'
    :param bool dnn_use_bn: _description_, defaults to False
    """