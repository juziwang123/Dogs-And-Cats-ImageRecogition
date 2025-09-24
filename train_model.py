import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 数据集路径
train_dir = r'C:\Users\chiju\PycharmProjects\RecogitionImage\train\train_organized'

# 数据预处理和增强
# rescale=1./255 将像素值从0-255缩放到0-1
# validation_split=0.2 自动将20%的数据用于验证集
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 加载训练集数据
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # 所有图片都调整为150x150像素
    batch_size=32,
    class_mode='binary',  # 二分类问题
    subset='training'
)

# 加载验证集数据
validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 打印一些信息
print(f"训练集图像数量: {train_generator.samples}")
print(f"验证集图像数量: {validation_generator.samples}")
print(f"类别映射: {train_generator.class_indices}")
# 构建模型
model = Sequential([
    # 第一层卷积
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    # 第二层卷积
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # 第三层卷积
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    # 展平层，用于连接全连接层
    Flatten(),
    # 全连接层
    Dense(512, activation='relu'),
    # 输出层
    Dense(1, activation='sigmoid')  # 二分类使用sigmoid激活函数
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 保存模型
model.save('dogs_vs_cats_model.h5')

# 绘制训练过程曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# train_model.py
# ...（之前的代码，包括训练模型和保存模型）



# 假设你的测试集图片在 dogs-vs-cats/test 文件夹中
# 如果你的测试集也按照 cats 和 dogs 文件夹组织，那么路径就是 dogs-vs-cats/test
test_dir = 'C:/Users/chiju/PycharmProjects/RecogitionImage/test/test'

# 创建一个不进行数据增强的生成器，只做缩放
test_datagen = ImageDataGenerator(rescale=1./255)

# 从测试目录加载数据
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 加载你训练好的模型
model = tf.keras.models.load_model('dogs_vs_cats_model.h5')

# 在测试集上评估模型
print("\n开始评估模型性能...")
loss, accuracy = model.evaluate(test_generator)

print(f"\n测试集损失 (Test Loss): {loss:.4f}")
print(f"测试集准确率 (Test Accuracy): {accuracy:.4f}")