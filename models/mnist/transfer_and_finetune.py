import numpy as np
import tensorflow as tf
import os

from models import MODELS_DIR
from models.mnist.dataset_reader import dataset_input_fn


def cnn_model_no_top(features, trainable):
    """
    :param features: 原始输入
    :param trainable: 该层的变量是否可训练
    :return: 不含最上层全连接层的模型
    """
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, trainable=trainable)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu, trainable=trainable)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, shape=[-1, 7 * 7 * 64])
    return pool2_flat


def cnn_model_fn(features, labels, mode, params):
    """
    用于构造estimator的model_fn
    :param features: 输入
    :param labels: 标签
    :param mode: 模式
    :param params: 用于迁移学习和微调训练的参数
        nb_classes
        transfer
        finetune
        checkpoints
        learning_rate
    :return: EstimatorSpec
    """
    logits_name = "predictions"
    # 把labels转换成ont-hot 形式
    labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params["nb_classes"])
    # 迁移学习不允许修改底层的参数
    model_no_top = cnn_model_no_top(features["x"], trainable=not (params.get("transfer") or params.get("finetune")))
    with tf.name_scope("finetune"):
        # 此层在第二次迁移学习时允许修改参数，将第二次迁移学习称作微调了
        dense = tf.layers.dense(inputs=model_no_top, units=1024, activation=tf.nn.relu, trainable=params.get("finetune"))
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # 最上层任何训练都可以修改参数
    logits = tf.layers.dense(inputs=dropout, units=params.get("nb_classes"), name=logits_name)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # 使用softmax交叉熵作为损失函数
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # 加载已有的存档点参数的方法
        if (params.get("transfer") or params.get("finetune")) and not params.get("checkpoint"):
            raise ValueError("must specify a checkpoint when transfer learning")
        if params.get("transfer") or params.get("finetune"):
            exclude = [logits_name] if params.get("transfer") else None
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(params.get("checkpoint"), {v.name.split(':')[0]: v for v in variables_to_restore})

        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=params.get("learning_rate", 0.0001))
        train_op = optimizer.minimize(loss, global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, 1),
                                        predictions=predictions['classes'],
                                        name='accuracy')
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


if __name__ == '__main__':
    # 训练MNIST的estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_model", params={
        "nb_classes": 10,
    })
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # 训练
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=np.asarray(train_labels),
        batch_size=50, num_epochs=50, shuffle=True
    )

    mnist_classifier.train(train_input_fn)
    # 验证
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=np.asarray(eval_labels),
        batch_size=50, num_epochs=2, shuffle=True
    )

    eval_result = mnist_classifier.evaluate(eval_input_fn)
    print(eval_result)  # MNIST数据集上的准确率

    # 第一次迁移学习，即只重新训练最上层的全连接层
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_transfer_model", params={
        "transfer": True,
        "nb_classes": 2,
        "checkpoints": os.path.join(MODELS_DIR, "models", "mnist", "mnist_model"),
    })

    train_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="train",
        epoch=100, batch_size=50,
        feature_name="x"
    )

    mnist_classifier.train(train_input_fn)

    eval_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="eval",
        epoch=1, batch_size=50,
        feature_name="x"
    )
    result = mnist_classifier.evaluate(eval_input_fn)
    print(result)

    # 第二次迁移学习，训练所有全连接层
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./mnist_finetune_model", params={
        "finetune": True,
        "nb_classes": 2,
        "checkpoints": os.path.join(MODELS_DIR, "models", "mnist", "mnist_transfer_model"),
    })

    train_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\训练集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="train",
        epoch=100, batch_size=50,
        feature_name="x"
    )

    mnist_classifier.train(train_input_fn)

    eval_input_fn = dataset_input_fn(
        folders=[
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample041"),  # e
            os.path.join("D:\\验证集\\Hnd\\Img", "Sample045"),  # i
        ],
        labels=[0, 1],
        height=28, width=28, channels=1,
        scope_name="eval",
        epoch=1, batch_size=50,
        feature_name="x"
    )
    result = mnist_classifier.evaluate(eval_input_fn)
    print(result)