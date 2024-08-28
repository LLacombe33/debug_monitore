import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import logging
import flask_monitoringdashboard as dashboard
import tensorflow as tf

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

dashboard.config.enable_logging = True
dashboard.bind(app)
dashboard.config.monitor_level = 3

MODEL_PATH = "models/unet_vgg16_categorical_crossentropy_raw_data.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    app.logger.info("Modèle chargé avec succès.")
except Exception as e:
    app.logger.error(f"Erreur lors du chargement du modèle: {e}")
    model = None


@app.route('/')
def index():
    app.logger.info("Chargement de index.html")
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500

    try:
        colors = np.array([[68, 1, 84],
                           [70,  49, 126],
                           [54,  91, 140],
                           [39, 126, 142],
                           [31, 161, 135],
                           [73, 193, 109],
                           [159, 217,  56],
                           [253, 231,  36]])

        if request.method == 'POST':
            image = request.files['file']

            if image.filename == '':
                return "Nom de fichier invalide"

            img = Image.open(image)
            IMAGE_SIZE = 512
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE),
                                     resample=Image.Resampling.NEAREST)
            img_resized = np.array(img_resized)

            img_resized = np.expand_dims(img_resized, 0)
            img_resized = img_resized / 255.

            predict_mask = model.predict(img_resized, verbose=0)
            predict_mask = np.argmax(predict_mask, axis=3)
            predict_mask = np.squeeze(predict_mask, axis=0)
            predict_mask = predict_mask.astype(np.uint8)
            predict_mask = Image.fromarray(predict_mask)
            predict_mask = predict_mask.resize((img.size[0], img.size[1]),
                                               resample=Image.Resampling.NEAREST)

            predict_mask = np.array(predict_mask)
            predict_mask = colors[predict_mask]
            predict_mask = predict_mask.astype(np.uint8)

            buffered_img = BytesIO()
            img.save(buffered_img, format="PNG")
            base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

            buffered_mask = BytesIO()
            mask_image = Image.fromarray(predict_mask)
            mask_image.save(buffered_mask, format="PNG")
            base64_mask = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

            app.logger.info("Image successfully processed and prediction completed.")

            return jsonify({'message': "predict ok",
                            "img_data": base64_img,
                            "mask_data": base64_mask})
        else:
            return "Invalid request method."

    except FileNotFoundError as fnf_error:
        app.logger.error(fnf_error)
        return jsonify({'error': str(fnf_error)}), 404

    except Exception as e:
        app.logger.error("Error occurred: %s", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
