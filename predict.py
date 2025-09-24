import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 加载你训练好的模型
model_path = 'dogs_vs_cats_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()


def predict_image(image_path):
    """
    对单张图片进行预测
    """
    # 定义模型要求的图片尺寸
    img_size = (150, 150)

    # 加载图片并调整尺寸
    try:
        img = load_img(image_path, target_size=img_size)
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
        return

    # 将图片转换为Numpy数组
    img_array = img_to_array(img)

    # 扩展维度以匹配模型输入要求 (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # 归一化像素值
    img_array /= 255.0

    # 进行预测
    prediction = model.predict(img_array)

    # 解释预测结果
    # 输出是0到1之间的概率值
    # 由于我们使用的是sigmoid激活函数，如果结果小于0.5，它倾向于类别0（猫），否则是类别1（狗）。
    if prediction[0] < 0.5:
        category = "猫"
        confidence = (1 - prediction[0][0]) * 100
    else:
        category = "狗"
        confidence = prediction[0][0] * 100

    print(f"预测结果：这是一只 {category}")
    print(f"置信度：{confidence:.2f}%")


if __name__ == "__main__":
    # 在这里替换为你要测试的图片路径
    test_image_path = 'C:/Users/chiju/PycharmProjects/RecogitionImage/test/test/1.jpg'

    # 调用预测函数
    predict_image(test_image_path)