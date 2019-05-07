import base64
import io
import os
import tempfile
import warnings

import lamps
from PIL import Image
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
application = Flask(__name__)
application.config['MAX_CONTENT_PATH'] = 32 * 1024 * 1024

@application.route('/')
def hello_world():
    return render_template('index.html')

@application.route('/about_factorio_lamps.html')
def about_lamps():
    return render_template('about_factorio_lamps.html')

@application.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(application.root_path, 'static'),
                               'favicon.ico',
                               mimetype='image/vnd.microsoft.icon')

MAX_LAMPS = 50000

@application.route('/factorio_lamps', methods=['GET', 'POST'])
def process_lamps():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' in request.form and request.form['image'] == 'cache':
            cache_filename = secure_filename(request.form['cachefile'])
            cache_dir = secure_filename(request.form['cachedir'])
            tmp_file = os.path.join(tempfile.gettempdir(), cache_dir, cache_filename)
            print('Attempting to reuse %s' % tmp_file)
            try:
                image = lamps.open_rotated_image(tmp_file)
            except FileNotFoundError:
                return render_template('lamp.html',
                                       error='Sorry, the cache file no longer exists')
            except OSError:
                return render_template('lamp.html',
                                       error='Sorry, the cache file has been corrupted')
        else:
            if 'file' not in request.files:
                return redirect(request.url)

            try:
                image = lamps.open_rotated_image(request.files['file'])
            except FileNotFoundError:
                return render_template('lamp.html',
                                       error='Upload error: image not found')
            except OSError:
                return render_template('lamp.html',
                                       error='Upload error: unable to read image file')

            filename = request.files['file'].filename
            filename = secure_filename(filename)
            cache_filename = os.path.split(filename)[1]
            # The form seems to dislike _ at the end of a field.  We
            # finesse that by making sure it always ends with 'lamp'
            tempdir = tempfile.mkdtemp(suffix='lamp')
            cache_dir = os.path.split(tempdir)[1]
            tmp_file = os.path.join(tempdir, cache_filename)
            print("Storing %s in %s" % (filename, tmp_file))
            request.files['file'].seek(0)
            request.files['file'].save(tmp_file)

        resize = request.form.get('resize', 'default')
        width = request.form.get('width', '90', type=int)
        height = request.form.get('height', '60', type=int)
        num_lamps = request.form.get('lamps', '5000', type=int)
        if resize == 'default':
            image = lamps.resize_image(image, default=True)
        elif resize == 'lamps':
            if num_lamps > MAX_LAMPS:
                return render_template('lamp.html',
                                       error='Unable to handle more than %d lamps' % MAX_LAMPS)
            image = lamps.resize_image(image, lamps=num_lamps)
        elif resize == 'size':
            if height * width > MAX_LAMPS:
                return render_template('lamp.html',
                                       error='Unable to handle more than %d lamps' % MAX_LAMPS)
            image = lamps.resize_image(image, shape=(width, height))
        else:
            return render_template('lamp.html',
                                   error='Unknown resize option')

        colors = request.form.get('colors', 'base')
        base_black = request.form.get('base_black', 'False') == 'True'
        if colors == 'expanded':
            color_set = lamps.EXPANDED_LAMP_COLORS
            disable_black = False
        elif colors == 'dectorio':
            color_set = lamps.DECTORIO_LAMP_COLORS
            disable_black = False
        elif colors == 'base':
            color_set = lamps.BASE_COLORS
            disable_black = True
            if not base_black:
                color_set = [x for x in color_set if x.name != 'signal-black']
        else:
            return render_template('lamp.html',
                                   error='Unknown color set')

        method = request.form.get('method', 'kmeans')
        if method == 'kmeans':
            bp, _ = lamps.convert_image_to_blueprint_kmeans(image, color_set, disable_black)
        elif method == 'nearest':
            bp, _ = lamps.convert_image_to_blueprint_nearest(image, color_set, disable_black)
        else:
            return render_template('lamp.html',
                                   error='Unknown color reduction method')

        preview_image = lamps.convert_blueprint_to_preview(bp, color_set)
        f = io.BytesIO()
        preview_image.save(f, format="PNG")
        preview = base64.b64encode(f.getvalue())

        stats = lamps.extract_blueprint_stats(bp)
        
        return render_template('lamp.html', bp=bp, 
                               cache_filename=cache_filename, cache_dir=cache_dir,
                               preview=preview.decode("utf-8"), stats=stats,
                               resize=resize, width=width, height=height,
                               num_lamps=num_lamps, colors=colors, method=method,
                               base_black=base_black)

    return render_template('lamp.html')

if __name__ == "__main__":
    #application.run()
    application.run(host="0.0.0.0", port=8080)

