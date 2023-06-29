from flask import Flask, request, render_template, jsonify
from machinelearning import processing, predict
import numpy as np

app = Flask(__name__)

shared_data = {
    'image_data': None,
    'model': None
}

@app.route('/')
def index():
    model = shared_data['model']
    print(model)
    print(shared_data)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def image():
    image_data = request.json.get('image')
    shared_data['image_data'] = image_data
    model = shared_data['model']
    if image_data == None:
        return render_template('index.html')
    image = processing.base64_to_PIL(image_data)
    image = processing.process_image(image, resize=True)
    if model == "TF_NN":
        pred = predict.tfnn_prediction(image)
    elif model == "CNN":
        pred = predict.cnn_prediction(image)
        print("CNN pred", pred)
    else:
        pred = 23
    return jsonify(prediction=int(pred))

@app.route('/model', methods=['POST'])
def model():
    model = request.json.get('selectedValue')
    shared_data['model'] = model
    return jsonify(model=model)
    # return {'message': 'Model request processed'}

if __name__ == '__main__':
    app.run(port=4000, debug=True)