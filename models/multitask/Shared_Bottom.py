import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input

def Shared_Bottom(dnn_feature_columns, num_tasks=None, task_types=None, task_names=None, 
    bottom_dnn_units=[128, 128], tower_dnn_units_lists=[[64, 32], [64, 32]], 
    l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """初始化Shared-Bottom多任务学习网络的框架

    :param _type_ dnn_feature_columns: 一个iterable，包含所有的特征，用于喂给模型的deep part
    :param _type_ num_tasks: 任务的个数，等同于输出的个数，必须大于1, defaults to None
    :param _type_ task_types: 各个任务的损失类型，例如binary/regression, defaults to None
    :param _type_ task_names: 各个任务的预估目标, defaults to None
    :param list bottom_dnn_units: 底层Shared-Bottom网络的layer的个数，以及每层layer的神经元个数, defaults to [128, 128]
    :param list tower_dnn_units_lists: 上层每个任务的网络的layer个数，以及每层layer的神经元个数，长度和num_tasks相同, defaults to [[64, 32], [64, 32]]
    :param float l2_reg_embedding: 应用到embedding向量上的L2正则程度, defaults to 0.00001
    :param int l2_reg_dnn: 应用到DNN网络的L2正则程度, defaults to 0
    :param int seed: 随机种子, defaults to 1024
    :param int dnn_dropout: DNN网络的dropout参数, defaults to 0
    :param str dnn_activation: 激活函数, defaults to 'relu'
    :param bool dnn_use_bn: 是否进行BN, defaults to False
    """

    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(task_types) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of task_types")

    for task_type in task_types:
        if task_type not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task_type))

    if num_tasks != len(tower_dnn_units_lists):
        raise ValueError("the length of tower_dnn_units_lists must be euqal to num_tasks")

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())