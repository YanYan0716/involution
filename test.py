import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.keras as keras
import config

if __name__ == '__main__':

    # schedules
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.BASE_LR,
        decay_rate=0.96,
        decay_steps=100,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()
    print('ok')

    BEST_ACC = 0
    # 训练和保存模型
    for epoch in range(config.START_EPOCH, config.TOTAL_EPOCHS):
        print(optimizer)
        lr_schedule.__call__(epoch)
