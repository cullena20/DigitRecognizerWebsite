const canvas = document.getElementById("drawingBoard");
const toolbar = document.getElementById("toolbar");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let isPainting = false;
let lineWidth = 25;
let startX;
let startY;

toolbar.addEventListener("click", e => {
    if (e.target.id === "clear") {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    if (e.target.id === "save") {
        const imageURL = canvas.toDataURL("/image/png");
        // Create a temporary <a> element
        const downloadLink = document.createElement("a");
        downloadLink.href = imageURL;
        
        // Set the filename for the download
        downloadLink.download = "my-drawing.png";
        
        // Trigger a click event on the download link
        downloadLink.click();
    }
    if (e.target.id === "predict") {
        const imageURL = canvas.toDataURL("/image/png");

        // Remove the data URL prefix (e.g., "data:image/png;base64,")
        const base64Data = imageURL.replace(/^data:image\/(png|jpeg);base64,/, '');

        // Send the image data to the server for processing
        fetch('/predict', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Data })
        })
        .then(response => response.json())
        .then(result => {
            // Handle the response from the server
            console.log(result);
            document.getElementById('prediction').textContent = result.prediction;
        })
        .catch(error => {
            // Handle any errors that occur during the request
            console.error('Error:', error);
        });
    }
});

toolbar.addEventListener("change", e => {
    if(e.target.id === "stroke") {
        ctx.strokeStyle = e.target.value;
    }

    if(e.target.id === "lineWidth") {
        lineWidth = e.target.value;
    }
});

const draw = e => {
    if(!isPainting) {
        return;
    }

    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";

    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
}

canvas.addEventListener("mousedown", e => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;
});

canvas.addEventListener("mouseup", e => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
});

canvas.addEventListener("mousemove", draw);2

const dropdown = document.getElementById('model-dropdown');
const defaultSelectedValue = dropdown.value;

 // Event listener for dropdown value changes
 dropdown.addEventListener('change', () => {
    const selectedValue = dropdown.value;
    sendDropdownValue(selectedValue);
});

// Function to send the dropdown value to the server
function sendDropdownValue(selectedValue) {
    fetch('/model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ selectedValue })
    })
    .then(response => response.json())
    .then(result => {
        // Handle the response from the server
        console.log(result);
        if (result.model == "TF_NN"){
            document.getElementById('description').textContent = `TF NN is a simple Neural Network trained in Tensorflow that
            achieves 97.7% test accuracy. It has an input layer of size 784, a hidden layer of size 128,
            and an output layer of size 10.`}
        else if (result.model == "MY_NN"){
            document.getElementById('description').textContent = "B"}
        else if (result.model == "CNN"){
            document.getElementById('description').textContent = `CNN is a Convolutional Neural Network trained in Tensorflow that achieves a 99.2% test accuracy.
            It's architecture is as follows: input layer of size (28, 28, 1), convolutional layer (32 kernels of size (3, 3)), max pooling layer (pool size (2, 2)), 
            dropout layer (dropout 0.25), convolutional layer (64 kernels of size (3, 3)), max pooling layer (pool size (2, 2)), 
            dropout layer (dropout 0.25), flatten and dense layer (128 units), dropout layer, output layer (10 output units).`}
    })
    .then(data => {
        // Handle the response from the server if needed
    })
    .catch(error => {
        // Handle any errors that occur during the request
    });
}

// Send the default selected value to the server when the page loads
sendDropdownValue(defaultSelectedValue);