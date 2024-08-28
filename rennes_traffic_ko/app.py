from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from keras.models import load_model
from src.get_data import GetData
from src.utils import create_figure, prediction_from_model 
import flask_monitoringdashboard as dashboard
import logging


app = Flask(__name__)

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

data_retriever = GetData(url="https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-du-trafic-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B")
data = data_retriever()

try:
    model = load_model('modl.h5')
    logging.info("Model loaded successfully.")
except OSError as e:
    logging.error("Failed to load model.h5: %s", str(e))
    model = None


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        try:
            fig_map = create_figure(data)
            graph_json = fig_map.to_json()

            selected_hour = request.form['hour']

            cat_predict = prediction_from_model(model, selected_hour)

            color_pred_map = {0: ["Prédiction : Libre", "green"],
                              1: ["Prédiction : Dense", "orange"],
                              2: ["Prédiction : Bloqué", "red"]}

            return render_template('index.html',
                                   graph_json=graph_json,
                                   text_pred=color_pred_map[cat_predict][0],
                                   color_pred=color_pred_map[cat_predict][1])
        except Exception as e:
            app.logger.error(f"Error in POST request: {e}")
            return "An error occurred", 500
    else:
        try:
            fig_map = create_figure(data)
            graph_json = fig_map.to_json

            return render_template('index.html', graph_json=graph_json)

        except Exception as e:
            app.logger.error(f"Error in POST request: {e}")
            return "An error occurred", 500


dashboard.config.enable_logging = True
dashboard.bind(app)
dashboard.config.monitor_level = 3
dashboard.config.threshold = 1000

if __name__ == '__main__':
    app.run(debug=True)
