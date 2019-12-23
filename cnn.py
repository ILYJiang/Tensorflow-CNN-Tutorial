# coding=utf-8

import numpy as np
import tensorflow as tf
import PictureProcess

# 模型文件路径
model_path = "model/image_model"


def read_data(train=False):
    datas = []
    labels = []
    fpaths = []
    # for fname in os.listdir(data_dir):
    #     fpath = os.path.join(data_dir, fname)
    #     fpaths.append(fpath)
    #     image = Image.open(fpath)
    #     data = np.array(image) / 255.0
    #     label = int(fname.split("_")[0])
    #     datas.append(data)
    #     labels.append(label)

    fp, da, la = PictureProcess.train_test(train)
    datas.extend(da)
    labels.extend(la)
    fpaths.extend(fp)

    datas = np.array(datas)
    labels = np.array(labels)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels


def process(train=False):
    fpaths, datas, labels = read_data(train)
    # 计算有多少类图片
    num_classes = len(set(labels))
    if not train:
        num_classes = 3
    print(num_classes)
    # 定义Placeholder，存放输入和标签
    datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_placeholder = tf.placeholder(tf.int32, [None])

    # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
    conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

    # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
    # 定义max-pooling层，pooling窗口为2x2，步长为2x2
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
    # 存放DropOut参数的容器，训练时为0.25，测试时为0
    dropout_placeholdr = tf.placeholder(tf.float32)
    # 将3维特征转换为1维向量
    flatten = tf.layers.flatten(pool1)
    # 全连接层，转换为长度为100的特征向量
    fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
    # 加上DropOut，防止过拟合
    dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)
    # 未激活的输出层
    logits = tf.layers.dense(dropout_fc, num_classes)

    # 用于保存和载入模型
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if train:

            print("训练模式")

            # 利用交叉熵定义损失
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(labels_placeholder, num_classes),
                logits=logits
            )
            # 平均损失
            mean_loss = tf.reduce_mean(losses)

            # 定义优化器，指定要优化的损失函数
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(losses)

            # 如果是训练，初始化参数
            sess.run(tf.global_variables_initializer())
            # 定义输入和Label以填充容器，训练时dropout为0.25
            train_feed_dict = {
                datas_placeholder: datas,
                labels_placeholder: labels,
                dropout_placeholdr: 0.25
            }
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels_placeholder, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            for step in range(500):
                _, mean_loss_val, acc = sess.run([optimizer, mean_loss, accuracy], feed_dict=train_feed_dict)

                if (step + 1) % 10 == 0:
                    print("step = {}\tmean loss = {}\t accuracy = {}".format(step + 1, mean_loss_val, acc))
            saver.save(sess, model_path)
            print("训练结束，保存模型到{}，本次训练图片数量:{}".format(model_path, len(labels)))
        else:
            print("测试模式")
            predicted_labels = tf.arg_max(logits, 1)
            # 如果是测试，载入参数
            print("从{}载入模型".format(model_path))
            saver.restore(sess, model_path)
            # label和名称的对照关系
            label_name_dict = {
                -1: "未知",
                0: "汽车",
                1: "罐车",
                2: "运渣车"
            }
            # 定义输入和Label以填充容器，测试时dropout为0
            test_feed_dict = {
                datas_placeholder: datas,
                labels_placeholder: labels,
                dropout_placeholdr: 0
            }

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.cast(labels_placeholder, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # 利用交叉熵定义损失
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(labels_placeholder, num_classes),
                logits=logits
            )
            # # 平均损失
            # mean_loss = tf.reduce_mean(losses)

            predicted_labels_val, loss_res, acc = sess.run([predicted_labels, losses, accuracy], feed_dict=test_feed_dict)
            print("本次测试图片数量:{}".format(len(labels)))

            print("预测返回转换后的结果:{}".format(predicted_labels_val))
            print("正确率:{}".format(acc))
            # 真实label与模型预测label
            for fpath, real_label, predicted_label, loss in zip(fpaths, labels, predicted_labels_val, loss_res):
                # 将label id转换为label名
                # if real_label != predicted_label:
                real_label_name = label_name_dict[real_label]
                predicted_label_name = label_name_dict[predicted_label]
                print("{}\t{} => {}, loss:{}".format(fpath, real_label_name, predicted_label_name, loss))


if __name__ == '__main__':
    # process(train=True)
    # process(train=False)
    # read_data(data_dir)
    print(np.log(10))
    print(np.log10(np.e))
    print(np.e)
