import os
from keras.applications import VGG16
import keras
from keras import backend as K
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator

# 安装 TensorFlow Privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import make_dp_model
from tensorflow_privacy.privacy.dp_query import gaussian_query

# 设置环境变量使用特定的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 加载预训练的 ResNet50 模型
resnet = keras.applications.resnet50.ResNet50(include_top=False,
                                              weights='imagenet',
                                              input_shape=(224, 224, 3))

# 冻结除最后4层外的所有层
for layer in resnet.layers[:-4]:
    layer.trainable = False

# 设置 Batch Normalization 层为非训练状态
for layer in resnet.layers:
    if layer.name.startswith('bn'):
        layer.call(layer.input, training=False)

# 创建新的模型
model = models.Sequential()
model.add(resnet)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

# 差分隐私优化器参数
learning_rate = 1e-4
noise_multiplier = 1.1  # 噪声乘数
l2_norm_clip = 1.0  # 梯度剪裁值

# 创建差分隐私优化器
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=batch_size,
    learning_rate=learning_rate
)

# 使用差分隐私模型
model = make_dp_model(model)

# 编译模型
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 数据预处理和生成器
train_data_dir = 'chest_xray/train'
validation_data_dir = 'chest_xray/test'
epochs = 10
batch_size = 64
img_height, img_width = 224, 224

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# 训练模型
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)
