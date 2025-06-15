import ternsorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input


def ESSM(dnn_feature_columns, task_type='binary', task_names=['ctr', 'ctcvr'], 
    tower_dnn_units_lists=[[128, 128], [128, 128]], l2_reg_embedding=0.00001, l2_reg_dnn=0,
    seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False):
    """实例化ESMM框架

    :param _type_ dnn_feature_columns: 一个iterable，包含所有的特征，用于喂给模型的deep part
    :param str task_type: 表示每个任务的损失函数类型, defaults to 'binary'
    :param list task_names: 表示每个任务的预估目标, defaults to ['ctr', 'ctcvr']
    :param list tower_dnn_units_lists: tower层的mlp层数，以及每层的神经元个数, defaults to [[128, 128], [128, 128]]
    :param float l2_reg_embedding: embedding向量的l2正则 defaults to 0.00001
    :param int l2_reg_dnn: dnn网络参数的l2正则, defaults to 0
    :param int seed: 随机数种子, defaults to 1024
    :param int dnn_dropout: dnn网络的dropout参数, defaults to 0
    :param str dnn_activation: dnn网络的激活参数, defaults to 'relu'
    :param bool dnn_use_bn: dnn网络是否使用bn, defaults to False
    """

    if len(task_names)!=2:
        raise ValueError("the length of task_names must be equal to 2")

    if len(tower_dnn_units_lists)!=2:
        raise ValueError("the length of tower_dnn_units_lists must be equal to 2")

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,seed)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    ctr_output = DNN(tower_dnn_units_lists[0], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    ctr_output = DNN(tower_dnn_units_lists[1], dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)

    ctr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(ctr_output)
    cvr_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(cvr_output)

    ctr_pred = PredictionLayer(task_type, name=task_names[0])(ctr_logit)
    cvr_pred = PredictionLayer(task_type)(cvr_logit)
    ctcvr_pred = tf.keras.layers.Multiply(name=task_names[1])([ctr_pred, cvr_pred])

    model = tf.keras.models.Model(inputs=inputs_list, outputs=[ctr_pred, cvr_pred])
    return model


if __name__ == "__main__":
    from utils import get_mtl_data
    dnn_feature_columns, train_model_input, test_model_input, y_list = get_mtl_data()
    model = ESSM(dnn_feature_columns, task_type='binary', task_names=['label_marital', 'label_income'], tower_dnn_units_lists=[[8],[8]])
    model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy"], metrics=['AUC'])
    history = model.fit(train_model_input, y_list, batch_size=256, epochs=5, verbose=2, validation_split=0.0)
