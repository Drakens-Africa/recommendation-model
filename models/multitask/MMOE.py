import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input, reduce_sum


def MMOE(dnn_feature_columns, num_tasks=None, task_types=None, task_names=None, num_experts=4,
    expert_dnn_units=[32, 32], gate_dnn_units=None, tower_dnn_units_lists=[[16, 8], [16, 8]],
    l2_reg_embedding=1e-5, l2_reg_dnn=0, seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """实例化MMOE多任务学习框架

    :param _type_ dnn_feature_columns: 一个iterable，包含所有的特征，用于喂给模型的deep part
    :param _type_ num_tasks: 任务的数量, defaults to None
    :param _type_ task_types: 每个任务的损失函数的类型, defaults to None
    :param _type_ task_names: 每个任务的预估目标, defaults to None
    :param int num_experts: 专家的数量, defaults to 4
    :param list expert_dnn_units: 专家网络的mlp层数以及每一层的神经元个数, defaults to [32, 32]
    :param _type_ gate_dnn_units: gate网络的mlpc层数以及每一层的神经元个数, defaults to None
    :param list tower_dnn_units_lists: tower层的mlp层数以及每一层的神经元个数, defaults to [[16, 8], [16, 8]]
    :param _type_ l2_reg_embedding: embedding向量的l2正则 defaults to 1e-5
    :param int l2_reg_dnn: dnn网络的l2正则, defaults to 0
    :param int seed: 随机数种子, defaults to 1024
    :param int dnn_dropout: dnn网络的dropout参数, defaults to 0
    :param str dnn_activation: dnn网络的激活函数, defaults to 'relu'
    :param bool dnn_use_bn: dnn网络是否使用bn, defaults to False
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

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # build expert layer
    expert_outs = []
    for i in range(num_experts):
        expert_network = DNN(expert_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='expert_'+str(i))(dnn_input)
        expert_outs.append(expert_network)
    expert_concat = tf.keras.layers.concatenate(expert_outs, axis=1, name='expert_concat')
    expert_concat = tf.keras.layers.Reshape([num_experts, expert_dnn_units[-1]], name='expert_reshape')(expert_concat)

    mmoe_outs = []
    for i in range(num_tasks): #one mmoe layer: nums_tasks = num_gates
        # build gate layers
        if gate_dnn_units is not None:
            gate_network = DNN(gate_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='gate_'+task_names[i])(dnn_input)
            gate_input = gate_network
        else:  # 在原论文中，gate就是一层dense layer之后再过softmax，所以这里不用再过DNN了
            gate_input = dnn_input
        gate_out = tf.keras.layers.Dense(num_experts, use_bias=False, activation='softmax', name='gate_softmax_'+task_names[i])(gate_input)
        gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

        # gate的权重和expert的输出相乘
        gate_mul_expert = tf.keras.layers.Multiply(name='gate_mul_expert_'+task_names[i])(expert_concat, gate_out)
        gate_mul_expert = tf.keras.layers.Lambda(lambda x: reduce_sum(x, axis=1, keep_dims=True))(gate_mul_expert)
        mmoe_outs.append(gate_mul_expert)

    task_outs = []
    for task_type, task_name, tower_dnn, mmoe_out in zip(task_types, task_names, tower_dnn_units_lists, mmoe_outs):
        # build tower layer
        tower_output = DNN(tower_dnn, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed, name='tower_'+task_name)(mmoe_out)
        
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outs)
    return model


if __name__ == "__main__":
    from utils import get_mtl_data
    dnn_feature_columns, train_model_input, test_model_input, y_list = get_mtl_data()

    model = MMOE(dnn_feature_columns, num_tasks=2, task_types=['binary', 'binary'], task_names=['income','marital'],
                num_experts=8, expert_dnn_units=[16], gate_dnn_units=None, tower_dnn_units_lists=[[8],[8]])
    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"], metrics=['AUC'])
    history = model.fit(train_model_input, y_list, batch_size=256, epochs=5, verbose=2, validation_split=0.0 )
