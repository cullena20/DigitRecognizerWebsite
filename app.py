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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    '''
    Gets the image data from the canvas where user's draw.
    Image is then processed and a prediction is made depending
    on the user's model selection.
    The prediction is then sent back to the server where JavaScript is
    used to display the prediction.
    This is triggered whenever the predict button is pressed.
    '''
    image_data = request.json.get('image')
    shared_data['image_data'] = image_data
    model = shared_data['model']
    if image_data == None:
        # if the user clicks predict without drawing anything
        return render_template('index.html')
    image = processing.base64_to_PIL(image_data)
    image = processing.process_image(image, resize=True)
    process_image_data = processing.PIL_to_base64(image)
    if model == "TF_NN":
        pred = predict.tfnn_prediction(image)
    elif model == "CNN":
        pred = predict.cnn_prediction(image)
    else:
        pred = -1
    return jsonify(prediction=int(np.argmax(pred)), image=process_image_data)

@app.route('/model', methods=['POST'])
def model():
    '''
    Gets the model chosen by a user under Model Selection.
    The model is stored in shared_data so that it can be accessed later
    to make a prediction.
    The model is then sent back to the server where JavaScript handles
    dynamic HTML for a model description.
    This is triggered when the page is first opened to get a default model,
    and whenever a user changes the model.
    '''
    model = request.json.get('selectedValue')
    shared_data['model'] = model
    return jsonify(model=model)

if __name__ == '__main__':
    app.run(port=4000, debug=True)