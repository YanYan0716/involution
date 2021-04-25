import os
import tensorflow as tf
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Model import Net
from DataAugment import read_image
from DataAugment import train_augment, train_normalize
from DataAugment import test_normalize, test_img_crop
from DataAugment import img_info


if __name__ == '__main__':
    # 超参数
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0005
    EPOCHS = 100
    MODEL_SAVE_PATH = './model/'
    BEST_ACC = 0
    BEST_LOSS = 10
    LOSS_PRINT = 30
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    NUM_CLASSES = 10  # 分类的种类

    # 加载数据集
    train_img_dir = ''
    train_file = './class/DS/train_small300/train2.txt'
    train_ds = img_info(train_file, train_img_dir)
    train_len = len(train_ds)
    train_ds = train_ds.map(read_image) \
        .map(train_normalize) \
        .map(train_augment) \
        .cache() \
        .shuffle(train_len) \
        .batch(batch_size=BATCH_SIZE) \
        .prefetch(AUTOTUNE)

    # print(len(train_ds)) # len(train_ds) = len(train_file) / batch_size
    test_img_dir = ''
    test_file = './class/DS/test/test_1.txt'
    test_ds = img_info(test_file, test_img_dir)
    test_ds = test_ds.map(read_image) \
        .map(test_normalize) \
        .batch(batch_size=BATCH_SIZE) \
        .prefetch(AUTOTUNE)

    # 定义模型
    net = Net(NUM_CLASSES)
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()

    print('ok')

    # 训练和保存模型
    for epoch in range(EPOCHS):
        print('training ...')
        LOSS_AVG = 0
        LOSS_SUM = 0
        for idx, (img, label) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                y_pred = net(img, training=True)
                loss = loss_fn(label, y_pred)

            gradients = tape.gradient(loss, net.trainable_weights)
            optimizer.apply_gradients(zip(gradients, net.trainable_weights))
            acc_metric.update_state(label, y_pred)
            LOSS_SUM += loss
            LOSS_AVG = LOSS_SUM / (idx + 1)
            if idx % LOSS_PRINT == 0:
                print(f'[{epoch}/{EPOCHS}] {idx} loss: {loss:.3f} / {LOSS_AVG:.3f}')

        print(f'[{epoch}/{EPOCHS}] train acc {train_acc:.3f}')
        train_acc = acc_metric.result()
        acc_metric.reset_states()

        # 每训练两次测试
        if epoch % 2 == 0:
            print('eval ...')
            for idx, (img, label) in enumerate(test_ds):
                with tf.GradientTape() as tape:
                    y_pred = net(img, training=False)
                    acc_metric.update_state(label, y_pred)

            test_acc = acc_metric.result()
            print(f'test acc {test_acc}, best acc {BEST_ACC}')
            acc_metric.reset_states()
            # 保存模型
            if test_acc > BEST_ACC:
                save_dir = os.path.join(MODEL_SAVE_PATH, str(epoch))
                net.save(save_dir)
                BEST_ACC = test_acc
