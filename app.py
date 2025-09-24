import os
import io
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# 定义模型路径
MODEL_PATH = 'dogs_vs_cats_model.h5'

# 加载模型
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型时出错: {e}")
    model = None

# 定义模型需要的图片尺寸
IMG_SIZE = (150, 150)


def predict_image(image_bytes):
    """
    接收图片字节流，处理并进行预测。
    """
    try:
        # 从字节流加载图片
        img = load_img(io.BytesIO(image_bytes), target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # 进行预测
        prediction = model.predict(img_array)

        # 解释预测结果
        if prediction[0][0] > 0.5:
            label = "狗"
            confidence = float(prediction[0][0])
        else:
            label = "猫"
            confidence = 1.0 - float(prediction[0][0])

        return label, confidence

    except Exception as e:
        return "预测失败", 0.0


@app.route('/')
def index():
    """
    渲染主页，这是用户访问网站时看到的页面。
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    处理图片上传和预测请求。
    """
    if 'file' not in request.files:
        return jsonify({"error": "没有文件部分"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400

    if file:
        # 读取文件内容
        image_bytes = file.read()

        # 检查模型是否加载成功
        if model is None:
            return jsonify({"error": "模型未加载，无法进行预测"}), 500

        # 进行预测
        label, confidence = predict_image(image_bytes)

        # 返回 JSON 格式的结果
        return jsonify({
            "label": label,
            "confidence": f"{confidence:.2%}"
        })


if __name__ == '__main__':
    # 在本地运行开发服务器
    app.run(debug=True)