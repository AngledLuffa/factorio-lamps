import base64
import io
import os
import warnings

import lamps
from PIL import Image
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
application = Flask(__name__)
application.config['MAX_CONTENT_PATH'] = 32 * 1024 * 1024

@application.route('/')
def hello_world():
    return render_template('index.html')

@application.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(application.root_path, 'static'),
                               'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')

@application.route('/factorio_lamps', methods=['GET', 'POST'])
def process_lamps():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        image = lamps.open_rotated_path(request.files['file'])

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
            colors = lamps.EXPANDED_LAMP_COLORS
            disable_black = False
        elif colors == 'dectorio':
            colors = lamps.DECTORIO_LAMP_COLORS
            disable_black = False
        else:   # include 'base' or undefined
            colors = lamps.BASE_COLORS
            disable_black = True
            if not bool(request.form.get('base_black', None)):
                colors = [x for x in colors if x.name != 'signal-black']

        bp, new_image = lamps.convert_image_to_blueprint_kmeans(image, colors, disable_black)

        preview_image = lamps.convert_blueprint_to_preview(bp, colors)
        f = io.BytesIO()
        preview_image.save(f, format="PNG")
        preview = base64.b64encode(f.getvalue())

        return render_template('lamp.html', bp=bp, preview=preview.decode("utf-8"))

    return render_template('lamp.html')

if __name__ == "__main__":
    #application.run()
    application.run(host="0.0.0.0", port=8080)

