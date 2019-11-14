# UNTUK MENDAPATKAN NAMA DIREKTORI DAN MENGGABUNGKAN PATH
from os.path import dirname, join

# UNTUK MENNAMBAHKAN AXIS, MEMUAT MODEL DAN GAMBAR
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing import image

# UNTUK BACK-END SITUS
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

# BIAR BISA JALAN WEB'NYA
app = Flask(__name__)

# MUAT MODEL YANG TELAH ANTUM LATIH
model = load_model("models/koceng_anjeng.h5")
model._make_predict_function()
print('Oke Siap!')

# FUNGSI UNTUK PRAPROSES CITRA HASIL UNGGAHAN
def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

# UNTUK MEMUAT LAMAN UTAMA/MERENDER index.html
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# UNTUK MENAMPILKAN HASIL PREDIKSI
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        
        basepath = dirname(__file__)
        file_path = join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        new_image = load_image(file_path)
        pred = model.predict(new_image)
        
        if pred[0][0] > pred[0][1]:
            return str("KOCHENG")
        else:
            return str("ANJENG")

    return None

# JALANKAN SERVER'NYA
if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
