# BUILT-IN LIBRARY UNTUK MENDAPATKAN NAMA DIREKTORI DAN MENGGABUNGKAN 
from os.path import dirname, join

from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing import image

from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

app = Flask(__name__)

# Load your trained model
model = load_model("models/koceng_anjeng.h5")
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = dirname(__file__)
        file_path = join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        new_image = load_image(file_path)
        pred = model.predict(new_image)

        result = str(pred[0])
        return result
    return None


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
