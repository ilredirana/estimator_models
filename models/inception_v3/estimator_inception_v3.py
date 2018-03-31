from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from models import BASE_DIR
from models.inception_v3.dataset_reader import dataset_input_fn


def conv2d_bn(inputs, filters, kernel_size, strides=(1, 1), padding="same", data_format="channels_last", name=None,
              trainable=True):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    conv2d = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                              strides=strides, padding=padding, data_format=data_format, name=conv_name,
                              trainable=trainable)
    batch_normalization = tf.layers.batch_normalization(inputs=conv2d, axis=3 if data_format == "channels_last" else 1,
                                                        scale=False, name=bn_name,
                                                        trainable=trainable)
    act = tf.nn.relu(batch_normalization, name=name)
    return act


def global_average_pool2d(inputs, name=None):
    return tf.reduce_mean(inputs, axis=[1, 2], name=name)


def global_max_pool2d(inputs, name=None):
    return tf.reduce_max(inputs, axis=[1, 2], name=name)


def inception_v3_no_top(features, trainable):
    input_layer = tf.reshape(features["images"], [-1, 299, 299, 3])
    x = conv2d_bn(inputs=input_layer, filters=32, kernel_size=3, strides=(2, 2), padding='valid', trainable=trainable)
    x = conv2d_bn(inputs=x, filters=32, kernel_size=3, padding='valid', trainable=trainable)
    x = conv2d_bn(inputs=x, filters=64, kernel_size=3, trainable=trainable)
    x = tf.layers.max_pooling2d(inputs=x, pool_size=(3, 3), strides=(2, 2))

    x = conv2d_bn(inputs=x, filters=80, kernel_size=1, padding='valid', trainable=trainable)
    x = conv2d_bn(inputs=x, filters=192, kernel_size=3, padding='valid', trainable=trainable)
    x = tf.layers.max_pooling2d(inputs=x, pool_size=(3, 3), strides=(2, 2))

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, trainable=trainable)

    branch5x5 = conv2d_bn(x, 48, 1, trainable=trainable)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, trainable=trainable)

    branch_pool = tf.layers.average_pooling2d(inputs=x, pool_size=(3, 3), strides=(1, 1), padding='same')
    branch_pool = conv2d_bn(branch_pool, 32, 1, trainable=trainable)
    x = tf.concat(values=[branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, trainable=trainable)

    branch5x5 = conv2d_bn(x, 48, 1, trainable=trainable)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, trainable=trainable)

    branch_pool = tf.layers.average_pooling2d(inputs=x, pool_size=(3, 3), strides=(1, 1), padding='same')
    branch_pool = conv2d_bn(branch_pool, 64, 1, trainable=trainable)
    x = tf.concat(values=[branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, trainable=trainable)

    branch5x5 = conv2d_bn(x, 48, 1, trainable=trainable)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, trainable=trainable)

    branch_pool = tf.layers.average_pooling2d(inputs=x, pool_size=(3, 3), strides=(1, 1), padding='same')
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, trainable=trainable)
    x = tf.concat(values=[branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(inputs=x, filters=384, kernel_size=3, strides=(2, 2), padding='valid', trainable=trainable)

    branch3x3dbl = conv2d_bn(x, 64, 1, trainable=trainable)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, trainable=trainable)
    branch3x3dbl = conv2d_bn(inputs=branch3x3dbl, filters=96, kernel_size=3, strides=(2, 2), padding='valid',
                             trainable=trainable)

    branch_pool = tf.layers.max_pooling2d(inputs=x, pool_size=(3, 3), strides=(2, 2))
    x = tf.concat([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, trainable=trainable)

    branch7x7 = conv2d_bn(x, 128, 1, trainable=trainable)
    branch7x7 = conv2d_bn(branch7x7, 128, (1, 7), trainable=trainable)
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1), trainable=trainable)

    branch7x7dbl = conv2d_bn(x, 128, 1, trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (7, 1), trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (1, 7), trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, (7, 1), trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7), trainable=trainable)

    branch_pool = tf.layers.average_pooling2d(inputs=x, pool_size=(3, 3), strides=(1, 1), padding='same')
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, trainable=trainable)
    x = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, trainable=trainable)

        branch7x7 = conv2d_bn(x, 160, 1, trainable=trainable)
        branch7x7 = conv2d_bn(branch7x7, 160, (1, 7), trainable=trainable)
        branch7x7 = conv2d_bn(branch7x7, 192, (7, 1), trainable=trainable)

        branch7x7dbl = conv2d_bn(x, 160, 1, trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (7, 1), trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (1, 7), trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, (7, 1), trainable=trainable)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7), trainable=trainable)

        branch_pool = tf.layers.average_pooling2d(inputs=x, pool_size=(3, 3), strides=(1, 1), padding='same')
        branch_pool = conv2d_bn(branch_pool, 192, 1, trainable=trainable)
        x = tf.concat(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, trainable=trainable)

    branch7x7 = conv2d_bn(x, 192, 1, trainable=trainable)
    branch7x7 = conv2d_bn(branch7x7, 192, (1, 7), trainable=trainable)
    branch7x7 = conv2d_bn(branch7x7, 192, (7, 1), trainable=trainable)

    branch7x7dbl = conv2d_bn(x, 192, 1, trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1), trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7), trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (7, 1), trainable=trainable)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, (1, 7), trainable=trainable)

    branch_pool = tf.layers.average_pooling2d(inputs=x, pool_size=(3, 3), strides=(1, 1), padding='same')
    branch_pool = conv2d_bn(branch_pool, 192, 1, trainable=trainable)
    x = tf.concat(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, trainable=trainable)
    branch3x3 = conv2d_bn(branch3x3, 320, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, trainable=trainable)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, (1, 7), trainable=trainable)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, (7, 1), trainable=trainable)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, strides=(2, 2), padding='valid')

    branch_pool = tf.layers.max_pooling2d(inputs=x, pool_size=(3, 3), strides=(2, 2))
    x = tf.concat(
        [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, trainable=trainable)

        branch3x3 = conv2d_bn(x, 384, 1, trainable=trainable)
        branch3x3_1 = conv2d_bn(branch3x3, 384, (1, 3), trainable=trainable)
        branch3x3_2 = conv2d_bn(branch3x3, 384, (3, 1), trainable=trainable)
        branch3x3 = tf.concat(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, trainable=trainable)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, (3, 3), trainable=trainable)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, (1, 3), trainable=trainable)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, (3, 1), trainable=trainable)
        branch3x3dbl = tf.concat(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = tf.layers.average_pooling2d(inputs=x, pool_size=(3, 3), strides=(1, 1), padding='same')
        branch_pool = conv2d_bn(branch_pool, 192, 1, trainable=trainable)
        x = tf.concat(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i))

    return global_average_pool2d(x, name='avg_pool')


def model_fn(features, labels, mode, params):
    """
    
    :param features: dict {"images" : Tensor(shape=(batch_size, 299, 299, 3))}
    :param labels: Tensor(shape=(batch_size,)), 需要将其转为one-hot形式
    :param mode: Estimator的类型
    :param params: 
        nb_classes: 类别数量，决定全连接层的units
        transfer: 第一次迁移学习，这里把它称为迁移
        finetune: 第二次迁移学习，这里把它称为微调
        checkpoint: 存档点路径
    :return: 
    """
    logits_name = "predictions"
    labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params["nb_classes"])
    no_top_model = inception_v3_no_top(features, trainable=not (params.get("transfer") or params.get("finetune")))
    with tf.name_scope("finetune"):
        # 允许第二次迁移学习时修改的参数，如果迁移学习的数据量大，可以考虑将前面的CNN层也允许training
        dense = tf.layers.dense(inputs=no_top_model, units=1024, activation=tf.nn.relu,
                                trainable=params.get("finetune"))
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # 最后一层全连接必然允许training
    predict_model = tf.layers.dense(inputs=dropout, units=params.get("nb_classes"), name=logits_name)
    predictions = {
        "classes": tf.argmax(input=predict_model, axis=1),
        "probabilities": tf.nn.softmax(predict_model, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=predict_model)
    if mode == tf.estimator.ModeKeys.TRAIN:
        if (params.get("transfer") or params.get("finetune")) and not params.get("checkpoint"):
            raise ValueError("must specify a checkpoint when transfer learning")
        if params.get("transfer") or params.get("finetune"):
            # 当训练方式为第一次迁移学习时，由于dense的units数量变化，不能加载该层的参数
            # 如果是第二次迁移学习，则可以在已经学习的基础上继续修改
            exclude = [logits_name] if params.get("transfer") else None
            # 不想使用slim的这个方法，考虑修改为tf核心库的ff
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            tf.train.init_from_checkpoint(params.get("checkpoint"),
                                          {v.name.split(':')[0]: v for v in variables_to_restore})

        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)  # 学习率应根据训练类型来设置，此处未作修改
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
    # 限制显存用量，先给0.8， 不够再增加
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session_config.gpu_options.allow_growth = True
    
    # 先是从头训练一个猫狗分类器
    dog_cat_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                                model_dir=os.path.join(BASE_DIR, "weights", "inception_v3", "cat_dog"),
                                                config=tf.estimator.RunConfig(session_config=session_config),
                                                params={"nb_classes": 2})
    
    # 输入的是这两个文件夹，分别是猫和狗， 它们的标签分别为0和1，训练100个epoch
    train_input_fn = dataset_input_fn(
        folders=[
            os.path.join(BASE_DIR, "images_dataset", "train", "cat"),
            os.path.join(BASE_DIR, "images_dataset", "train", "dog"),
        ],
        labels=[0, 1],
        height=299, width=299, channels=3,
        scope_name="train",
        epoch=100, batch_size=20,
        feature_name="images"
    )

    # 开始训练，没有指定steps或max_steps则会把100个epoch全部训练完才停止
    dog_cat_classifier.train(train_input_fn)

    # 验证的输入
    eval_input_fn = dataset_input_fn(
        folders=[
            os.path.join(BASE_DIR, "images_dataset", "eval", "cat"),
            os.path.join(BASE_DIR, "images_dataset", "eval", "dog"),
        ],
        labels=[0, 1],
        height=299, width=299, channels=3,
        scope_name="eval",
        epoch=1, batch_size=20,
        feature_name="images"
    )
    
    # 验证结果
    result = dog_cat_classifier.evaluate(eval_input_fn)
    print(result)

    # 迁移学习
    horse_sheep_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(BASE_DIR, "weights", "inception_v3", "horse_sheep"),
        params={
            "transfer": True,
            "nb_classes": 2,
            "checkpoint": os.path.join(BASE_DIR, "weights", "inception_v3", "cat_dog")
        })
    train_input_fn = dataset_input_fn(
        folders=[
            os.path.join(BASE_DIR, "images_dataset", "train", "horse"),
            os.path.join(BASE_DIR, "images_dataset", "train", "sheep"),
        ],
        labels=[0, 1],
        height=299, width=299, channels=3,
        scope_name="train",
        epoch=10, batch_size=50,
        feature_name="images"
    )

    horse_sheep_classifier.train(train_input_fn)

    eval_input_fn = dataset_input_fn(
        folders=[
            os.path.join(BASE_DIR, "images_dataset", "eval", "horse"),
            os.path.join(BASE_DIR, "images_dataset", "eval", "sheep"),
        ],
        labels=[0, 1],
        height=299, width=299, channels=3,
        scope_name="eval",
        epoch=1, batch_size=50,
        feature_name="images"
    )
    result = horse_sheep_classifier.evaluate(eval_input_fn)
    print(result)

    # 迁移学习第二部，开放更多的层允许训练
    horse_sheep_classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=os.path.join(BASE_DIR, "weights", "inception_v3", "horse_sheep"),
        params={
            "finetune": True,
            "nb_classes": 2,
            "checkpoint": os.path.join(BASE_DIR, "weights", "inception_v3", "horse_sheep")
        })
    train_input_fn = dataset_input_fn(
        folders=[
            os.path.join(BASE_DIR, "images_dataset", "train", "horse"),
            os.path.join(BASE_DIR, "images_dataset", "train", "sheep"),
        ],
        labels=[0, 1],
        height=299, width=299, channels=3,
        scope_name="train",
        epoch=10, batch_size=50,
        feature_name="images"
    )

    horse_sheep_classifier.train(train_input_fn)

    eval_input_fn = dataset_input_fn(
        folders=[
            os.path.join(BASE_DIR, "images_dataset", "eval", "horse"),
            os.path.join(BASE_DIR, "images_dataset", "eval", "sheep"),
        ],
        labels=[0, 1],
        height=299, width=299, channels=3,
        scope_name="eval",
        epoch=1, batch_size=50,
        feature_name="images"
    )
    result = horse_sheep_classifier.evaluate(eval_input_fn)
    print(result)
