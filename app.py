from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            tempo=float(request.form.get('tempo')),
            dynamics_range=float(request.form.get('dynamics_range')),
            vocal_presence=float(request.form.get('vocal_presence')),
            percussion_strength=float(request.form.get('percussion_strength')),
            string_instrument_detection=float(request.form.get('string_instrument_detection')),
            electronic_element_presence=float(request.form.get('electronic_element_presence')),
            rhythm_complexity=float(request.form.get('rhythm_complexity')),
            drums_influence=float(request.form.get('drums_influence')),
            distorted_guitar=float(request.form.get('distorted_guitar')),
            metal_frequencies=float(request.form.get('metal_frequencies')),
            ambient_sound_influence=float(request.form.get('ambient_sound_influence')),
            instrumental_overlaps=float(request.form.get('instrumental_overlaps'))
        )

        input_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()

        prediction = predict_pipeline.predict(input_df)

        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)