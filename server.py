from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import re
import base64
from io import BytesIO

app = Flask(__name__)

# モデルのロードと予測処理
def predict_katakana(image):
    # モデルをロードする
    loaded_model = tf.keras.models.load_model('trained_model.h5')

    # 画像のリサイズとモノクロ化
    resized_image = image.resize((28, 28))
    grayscale_image = Image.new("L", resized_image.size, color=255)
    grayscale_image.paste(resized_image, (0, 0), resized_image)

    # 画像をモデルの入力形式に変換
    image_array = np.array(grayscale_image) / 255.0  # 0-255の範囲を0-1に正規化
    input_data = image_array.reshape(1, 28, 28, 1)  # バッチサイズ1として形状を整える

    # 画像の認識を行う
    predictions = loaded_model.predict(input_data)

    # 予測結果に対応する文字列を返す
    predicted_class = np.argmax(predictions)
    if predicted_class == 0:
        return {"predicted_class": "ア"}
    elif predicted_class == 1:
        return {"predicted_class": "イ"}
    elif predicted_class == 2:
        return {"predicted_class": "ウ"}
    elif predicted_class == 3:
        return {"predicted_class": "エ"}
    elif predicted_class == 4:
        return {"predicted_class": "オ"}
    elif predicted_class == 5:
        return {"predicted_class": "カ"}
    elif predicted_class == 6:
        return {"predicted_class": "キ"}
    elif predicted_class == 7:
        return {"predicted_class": "ク"}
    elif predicted_class == 8:
        return {"predicted_class": "ケ"}
    elif predicted_class == 9:
        return {"predicted_class": "コ"}
    elif predicted_class == 10:
        return {"predicted_class": "サ"}
    elif predicted_class == 11:
        return {"predicted_class": "シ"}
    elif predicted_class == 12:
        return {"predicted_class": "ス"}
    elif predicted_class == 13:
        return {"predicted_class": "セ"}
    elif predicted_class == 14:
        return {"predicted_class": "ソ"}
    else:
        return {"predicted_class": "?"}

@app.route("/")
def index():
    return render_template("index.html", result=None)

@app.route("/recognize", methods=["POST"])
def recognize():
    image_data = re.sub('^data:image/.+;base64,', '', request.json['image'])
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # ここで画像の保存や前処理を行い、認識処理を呼び出します
    prediction = predict_katakana(image)

    # int64型のオブジェクトをintにキャストする
    prediction["predicted_class"] = (prediction["predicted_class"])


    return jsonify(prediction)
def reset():
    return jsonify({"result": "reset"})

if __name__ == "__main__":
    app.run()
