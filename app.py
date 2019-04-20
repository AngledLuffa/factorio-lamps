import base64
import io
import lamps
from PIL import Image
from flask import Flask, flash, redirect, render_template, request, url_for
app = Flask(__name__)
app.config['MAX_CONTENT_PATH'] = 32 * 1024 * 1024

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/foo')
def foo():
    return 'Hello, World!'

@app.route('/factorio_lamps', methods=['GET', 'POST'])
def process_lamps():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No image uploaded')
            return redirect(request.url)

        image = Image.open(request.files['file'])
        bp = lamps.convert_image_to_blueprint(image, (90, 60), False, 7)

        preview_image = lamps.convert_blueprint_to_preview(bp)
        f = io.BytesIO()
        preview_image.save(f, format="PNG")
        preview = base64.b64encode(f.getvalue())

        return render_template('lamp.html', bp=bp, preview=preview.decode("utf-8"))

    return render_template('lamp.html')

if __name__ == "__main__":
    #app.run()
    app.run(host="0.0.0.0", port=8080)

