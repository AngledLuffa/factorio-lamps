import base64
import json
import sys
import zlib

import numpy as np
from scipy.cluster.vq import kmeans2
from PIL import Image

def compress_blueprint(blueprint):
    blueprint = json.dumps(blueprint).encode("utf-8")
    blueprint = zlib.compress(blueprint)
    blueprint = base64.b64encode(blueprint)
    blueprint = blueprint.decode("utf-8")
    return "0" + blueprint

def decompress_blueprint(blueprint):
    # https://wiki.factorio.com/Blueprint_string_format
    # cut off the leading 0
    blueprint = blueprint[1:]
    blueprint = base64.b64decode(blueprint)
    blueprint = zlib.decompress(blueprint)
    return json.loads(blueprint)

def default_colors(K):
    defaults = ["signal-red", "signal-green", "signal-blue", "signal-yellow",
                "signal-pink", "signal-cyan", "signal-white"]
    if K == 7:
        return defaults
    elif K == 6:
        return defaults[:-1]
    else:
        raise RuntimeError("K must be 6 or 7")

def build_combinator(entity_number, x, y, color, color_lamp):
    combinator = {
        "entity_number": entity_number,
        "name": "constant-combinator",
        "position": {
            "x": x,
            "y": y
        },
        "direction": 6,
        "control_behavior": {
            "filters": [
                {
                    "signal": {
                        "type": "virtual",
                        "name": color
                    },
                    "count": 1,
                    "index": 1
                }
            ]
        },
        "connections": {
            "1": {
                "green": [
                    {
                        "entity_id": color_lamp
                    }
                ]
            }
        }
    }
    return combinator

def build_lamp(entity_number, x, y, color_combinator):
    lamp = {
        "entity_number": entity_number,
        "name": "small-lamp",
        "position": {
            "x": x,
            "y": y
        },
        "control_behavior": {
            "circuit_condition": {
                "first_signal": {
                    "type": "virtual",
                    "name": "signal-anything"
                },
                "constant": 0,
                "comparator": ">"
            },
            "use_colors": True
        },
        "connections": {
            "1": {
                "green": [
                    {
                        "entity_id": color_combinator
                    }
                ]
            }
        }
    }
    return lamp

def convert_to_blueprint(centroids, labels, width, height):
    blueprint = {
        "blueprint": {
            "icons": [
                {
                    "signal": {
                        "type": "item",
                        "name": "small-lamp"
                    },
                    "index": 1
                }
            ],
            "item": "blueprint"
        },
    }

    # TODO: arrange the centroids to be closest to the 7 colors
    # Can do this with mincost flow, for example
    # TODO: maybe include black?
    # TODO: maybe *don't* include white?
    label_to_colors = default_colors()
    
    labels = labels.reshape((height, width))
    entities = []

    lamps = {}

    for i in range(height):
        for j in range(width):
            neighbor = None
            if i > 0 and labels[i-1, j] == labels[i, j]:
                neighbor = lamps[(i-1, j)]
            elif j > 0 and labels[i, j-1] == labels[i, j]:
                neighbor = lamps[(i, j-1)]

            if neighbor:
                lamp = build_lamp(len(entities) + 1,
                                  j * 2 - width + 1,
                                  i * 2 - height,
                                  neighbor["entity_number"])
                lamps[(i, j)] = lamp
                connection = {"entity_id": lamp["entity_id"]}
                neighbor["connections"]["1"]["green"].append(connection)
                entities.append(lamp)
            else:
                color = label_to_colors[labels[i, j]]
                combinator = build_combinator(len(entities) + 1,
                                              j * 2 - width,
                                              i * 2 - height,
                                              color,
                                              len(entities) + 2)
                lamp = build_lamp(len(entities) + 2,
                                  j * 2 - width + 1,
                                  i * 2 - height,
                                  len(entities) + 1)
                lamps[(i, j)] = lamp
                entities.append(combinator)
                entities.append(lamp)

    # add enough poles to cover the image
    pole_x = list(range(-width+3, width-2, 7))
    if pole_x[-1] < width - 3:
        pole_x.append(width - 3)
    pole_y = list(range(-height+3, height-2, 6))
    if pole_y[-1] < height - 3:
        pole_y.append(height - 3)

    for i in pole_x:
        for j in pole_y:
            pole = {
                "entity_number": len(entities) + 1,
                "name": "medium-electric-pole",
                "position": {
                    "x": i,
                    "y": j
                }
            }
            entities.append(pole)

    blueprint["blueprint"]["entities"] = entities
    
    return blueprint

def convert_image_to_blueprint(image, shape, show_intermediates,
                               clusters):
    print("Original image size: %s" % str(image.size))
    if show_intermediates:
        image.show()
    if shape:
        width, height = shape
        print("Resizing to %d, %d" % (width, height))
        image = image.resize((width, height))
    else:
        width, height = image.size

    flat_image = np.asarray(image, dtype=np.float32)
    if flat_image.shape[2] != 3:
        raise RuntimeError("Only works on RGB images")
    flat_image = flat_image.reshape((width * height, 3))

    centroids, labels = kmeans2(flat_image, clusters,
                                iter=50, minit='points')

    # centroids will be a Kx3 array representing colors
    # labels will be which centroid for each pixel
    # so centroids[labels] will be the pixels mapped to their K colors
    flat_kmeans_image = centroids[labels]
    kmeans_image = flat_kmeans_image.reshape((height, width, 3))
    kmeans_image = np.array(kmeans_image, dtype=np.int8)

    new_image = Image.fromarray(kmeans_image, "RGB")
    if show_intermediates:
        new_image.show()

    blueprint = convert_to_blueprint(centroids, labels, width, height)
    return compress_blueprint(blueprint) 
    
        
if __name__ == '__main__':
    path = sys.argv[1]

    image = Image.open(path)
    if len(sys.argv) > 2:
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        shape = (width, height)
    else:
        shape = None

    bp = convert_image_to_blueprint(image, shape, False, 7)
    print
    print("BLUEPRINT")
    print(bp)
