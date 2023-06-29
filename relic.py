from flask import Flask, request, render_template, jsonify
from machinelearning import processing, predict
import numpy as np

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        model = request.json.get('selectedValue')
        # print(request.json)
        print("MODEL:", model)
        image_data = request.json.get('image')
        # print(image_data)
        if image_data == None:
            print("THERES NOTHING")
            return render_template('index.html')
        print("MADE IT HERE")
        image = processing.base64_to_PIL(image_data)
        image = processing.process_image(image, resize=True)
        if model == "TF_NN":
            pred = predict.tfnn_prediction(image)
        else:
            pred = 23
        return jsonify(prediction=int(pred))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=4000, debug=True)