import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd


from Rednet import RedNet
from Dataset import train_image, test_image
import config


if __name__ == '__main__':

    # 加载数据集
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    df_label = pd.read_csv(config.TRAIN_PATH)
    file_paths = df_label['name'].values
    labels = df_label['label'].values
    ds_label_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_label_train = ds_label_train\
        .map(train_image, num_parallel_calls=AUTOTUNE)\
        .shuffle(config.SHUFFLE_BUFFER)\
        .batch(config.BATCH_SIZE)

    df_test = pd.read_csv(config.TEST_PATH)
    file_test = df_test['file_name'].values
    labels_test = df_test['label'].values
    ds_label_test = tf.data.Dataset.from_tensor_slices((file_test, labels_test))
    ds_label_test = ds_label_test \
        .map(test_image, num_parallel_calls=AUTOTUNE) \
        .batch(config.BATCH_SIZE)

    # 定义模型
    net = RedNet(depth=26).model()

    # schedules
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=config.BASE_LR,
    #     decay_rate=0.96,
    #     decay_steps=5000,
    # )
    optimizer = keras.optimizers.Adam(lr=config.BASE_LR)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()
    print('ok')

    BEST_ACC = 0
    # 训练和保存模型
    for epoch in range(config.START_EPOCH, config.TOTAL_EPOCHS):
        print('training ...')
        LOSS_AVG = 0
        LOSS_SUM = 0
        for idx, (img, label) in enumerate(ds_label_train):
            with tf.GradientTape() as tape:
                y_pred = net(img, training=True)
                loss = loss_fn(label, y_pred)

            gradients = tape.gradient(loss, net.trainable_weights)
            optimizer.apply_gradients(zip(gradients, net.trainable_weights))
            acc_metric.update_state(label, y_pred)
            LOSS_SUM += loss
            LOSS_AVG = LOSS_SUM / (idx + 1)
            if idx % config.LOG_LOSS == 0:
                print(f'[{epoch:4d}/{config.TOTAL_EPOCHS:4d}] {idx:4d} loss: {loss:.3f} / {LOSS_AVG:.3f}')

        train_acc = acc_metric.result()
        print(f'[{epoch:4d}/{config.TOTAL_EPOCHS:4d}] train acc {train_acc:.3f}')
        acc_metric.reset_states()

        # 每训练两次测试
        if epoch % 2 == 0:
            print('eval ...')
            for idx, (img, label) in enumerate(ds_label_test):
                with tf.GradientTape() as tape:
                    y_pred = net(img, training=False)
                    acc_metric.update_state(label, y_pred)

            test_acc = acc_metric.result()
            print(f'test acc {test_acc:.3f}, best acc {BEST_ACC:.3f}')
            acc_metric.reset_states()
            # 保存模型
            if test_acc > BEST_ACC:
                save_dir = os.path.join(config.SAVE_PATH, str(epoch))
                net.save(save_dir)
                BEST_ACC = test_acc
