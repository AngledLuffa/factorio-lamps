import base64
import io
import os

import lamps
from PIL import Image
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
app = Flask(__name__)
app.config['MAX_CONTENT_PATH'] = 32 * 1024 * 1024

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')

@app.route('/factorio_lamps', methods=['GET', 'POST'])
def process_lamps():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        image = Image.open(request.files['file'])

        resize = request.form.get('resize', 'default')
        if resize == 'default':
            image = lamps.resize_image(image, default=True)
        elif resize == 'lamps':
            try:
                num_lamps = int(request.form.get('lamps', '5000'))
            except ValueError:
                num_lamps = 5000
            image = lamps.resize_image(image, lamps=num_lamps)
        elif resize == 'size':
            try:
                width = int(request.form.get('width', '90'))
            except ValueError:
                width = 90
            try:
                height = int(request.form.get('height', '60'))
            except ValueError:
                height = 60
            image = lamps.resize_image(image, shape=(width, height))

        colors = request.form.get('color', 'base')
        if colors == 'expanded':
            color_map = lamps.EXPANDED_LAMP_COLORS
            colors_used = color_map.keys()
            disable_black = False
        elif colors == 'dectorio':
            color_map = lamps.DECTORIO_LAMP_COLORS
            colors_used = color_map.keys()
            disable_black = False
        else:   # include 'base' or undefined
            color_map = lamps.BASE_COLORS
            colors_used = color_map.keys()
            disable_black = True
            if not bool(request.form.get('base_black', None)):
                colors_used = list(colors_used)
                colors_used.remove('signal-black')

        bp, new_image = lamps.convert_image_to_blueprint_kmeans(image, colors_used, color_map, disable_black)

        preview_image = lamps.convert_blueprint_to_preview(bp, color_map)
        f = io.BytesIO()
        preview_image.save(f, format="PNG")
        preview = base64.b64encode(f.getvalue())

        return render_template('lamp.html', bp=bp, preview=preview.decode("utf-8"))

    return render_template('lamp.html')

if __name__ == "__main__":
    #app.run()
    app.run(host="0.0.0.0", port=8080)

