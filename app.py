from flask import Flask, render_template, request, send_file
import os
import uuid
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

app = Flask(__name__)
UPLOAD_FOLDER = 'static/compressed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compress(img: Image.Image, img_quality: float):
    img = img.convert('RGB')
    img_np = np.array(img) / 255.0  
    compressed = []
    for i in range(3):
        imgsize = img_np[:, :, i] 
        pca = PCA(img_quality / 100.0) # image quality dynammic
        transformed = pca.fit_transform(imgsize)
        inverse_img = pca.inverse_transform(transformed)
        compressed.append(inverse_img)
    compressed_img = np.stack(compressed, axis=2)
    compressed_img = np.clip(compressed_img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(compressed_img)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        quality = float(request.form['quality'])

        if file and 1 <= quality <= 100:
            img = Image.open(file)
            compressed_img = compress(img, quality)
            filename = f"compressed_{uuid.uuid4().hex}.png"
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            compressed_img.save(save_path)
            return render_template('download.html', file=filename)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)