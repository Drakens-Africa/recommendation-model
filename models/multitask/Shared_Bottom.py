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

    """
    build_input_features的主要作用是把每个特征都转成一个个Input类。
    注意，这里每个特征也都是类。如果当前特征是SparseFeat，就转成1维的Input；如果是DenseFeat，就转成指定维数（在该特征类里面有指定）的Input。
    """
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    """
    input_from_feature_columns里面包含的步骤比较多。
    1. 获取embedding_matrix_dict。首先创建一个embedding表，包含所有的sparse和varlen sparse特征。embedding表是一个字典，其中key是该特征类里面的embedding_name，value是一个tf的Embedding类。
    2. 获取group_sparse_embedding_dict。把features里面所有的sparse特征取出来，并且做hash，然后去embedding表里面去查对应的embedding向量。
    3. 获取dense_value_list。把features里面所有dense特征的值，直接取出来/或者经过transform_fn再取出来。
    4. 获取sequence_embed_dict。把features里面所有varlen sparse特征取出来，并且做hash，然后去embedding表里面去查对应的embedding向量。注意，这里的sequence_embed_dict，key是每一个varlen sparse的特征名，value则是多个相同维度的embedding向量组成的二维向量，可以理解成序列特征。
    5. 获取group_varlen_sparse_embedding_dict。需要对上一步得到的sequence_embed_dict做polling，这里又分为两步。
        5.1 如果该特征为带权特征，那么需要先对序列中的每个item乘以对应的权重。注意，这里权重是提前定义好的，是放在特征类里面的，不是神经网络生成的权重。
        5.2 将上一步得到的序列特征再去做pooling，主要是max/sum/mean这几种方式
    6. 获取group_embedding_dict。合并group_sparse_embedding_dict和group_varlen_sparse_embedding_dict。
    7. 最终返回的是group_embedding_dict和dense_value_list
    """
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    shared_bottom_output = DNN(bottom_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    tasks_output = []
    for task_type, task_name, tower_dnn in zip(task_types, task_names, tower_dnn_units_list):
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(shared_bottom_output)

        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        tasks_output.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=tasks_output)
    return model


if __name__ == "__main__":
    from utils import get_mtl_data
    dnn_feature_columns, train_model_input, test_model_input, y_list = get_mtl_data()
    model = Shared_Bottom(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], tasks_name=['label_income', 'label_marital'], bottom_dnn_units=[16], tower_dnn_units_lists=[[8], [8]])
    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"], metrics=['AUC'])
    history = model.fit(train_model_input, y_list, batch_size=256, epochs=5, verbose=2, validation_split=0.0)