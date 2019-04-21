import base64
import json
import sys
import zlib

from collections import deque, OrderedDict

import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
from PIL import Image

BASE_COLORS_NO_WHITE = {
    "signal-red": np.array((255, 0, 0)),
    "signal-green": np.array((0, 255, 0)),
    "signal-blue": np.array((0, 0, 255)),
    "signal-yellow": np.array((255, 255, 0)),
    "signal-pink": np.array((255, 0, 255)),
    "signal-cyan": np.array((0, 255, 255)),
}

BASE_COLORS = dict(BASE_COLORS_NO_WHITE)
BASE_COLORS["signal-white"] = np.array((255, 255, 255))


def compress_blueprint(blueprint):
    """
    Convert the given blueprint to factorio's text format.

    https://wiki.factorio.com/Blueprint_string_format
    """
    blueprint = json.dumps(blueprint).encode("utf-8")
    blueprint = zlib.compress(blueprint)
    blueprint = base64.b64encode(blueprint)
    blueprint = blueprint.decode("utf-8")
    return "0" + blueprint

def decompress_blueprint(blueprint):
    """
    Decompresses a blueprint.

    Works for any blueprint, actually.

    https://wiki.factorio.com/Blueprint_string_format
    """
    # cut off the leading 0
    blueprint = blueprint[1:]
    blueprint = base64.b64decode(blueprint)
    blueprint = zlib.decompress(blueprint)
    return json.loads(blueprint)

def default_colors(K):
    if K == 7:
        return BASE_COLORS.keys()
    elif K == 6:
        return BASE_COLORS_NO_WHITE.keys()
    else:
        raise RuntimeError("K must be 6 or 7")

def min_cost_colors(centroids):
    """
    Assign colors based on min cost flow from centroid to color.
    """
    K = len(centroids)
    
    if K == 7:
        colors = BASE_COLORS
    elif K == 6:
        colors = BASE_COLORS_NO_WHITE
    elif K != 6:
        raise RuntimeError("K must be 6 or 7")

    # We build a mincost flow as follows:
    # source: K output
    # edges to each color: cost 0, flow 1
    # edges from each color to each centroid: cost L2, flow 1
    # edges from each centroid to sink: cost 0, flow 1
    # sink: K input

    centroid_names = ["C%d" % i for i in range(len(centroids))]
    
    G = nx.DiGraph()
    for c in centroid_names:
        G.add_node(c, demand=-1)
    for k in colors.keys():
        G.add_node(k, demand=1)
    for k in colors.keys():
        for c in range(len(centroids)):
            distance = int(np.linalg.norm(colors[k] - centroids[c]))
            G.add_edge(centroid_names[c], k, capacity=1, weight=distance)

    flow = nx.algorithms.min_cost_flow(G)
    flow_colors = []
    for source in sorted(flow.keys()):
        c = None
        for dest in flow[source]:
            if flow[source][dest] > 0:
                if c:
                    # TODO: can this happen if floating point result?
                    raise RuntimeError("Multiple colors mapped to same source")
                flow_colors.append(dest)
    return flow_colors
                
def build_combinator(entity_number, x, y, color):
    """
    Creates a constant combinator.

    The combinator emits the given color signal.
    """
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
        }
    }
    return combinator

def build_lamp(entity_number, x, y):
    """
    Builds a lamp blueprint.

    """
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
        }
    }
    return lamp

def add_connection(e1, e2):
    e2n = e2["entity_number"]
    if "connections" not in e1:
        e1["connections"] = { "1": { "green": [] } }
    e1["connections"]["1"]["green"].append({"entity_id": e2n})

def add_bidirectional_connection(e1, e2):
    add_connection(e1, e2)
    add_connection(e2, e1)    

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

    # TODO: maybe include black?
    # TODO: maybe *don't* include white?
    # TODO: make mincost/default an option
    #label_to_colors = default_colors(len(centroids))
    label_to_colors = min_cost_colors(centroids)
    
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
                                  i * 2 - height)
                lamps[(i, j)] = lamp
                add_bidirectional_connection(lamp, neighbor)
                entities.append(lamp)
            else:
                color = label_to_colors[labels[i, j]]
                combinator = build_combinator(len(entities) + 1,
                                              j * 2 - width,
                                              i * 2 - height,
                                              color)
                lamp = build_lamp(len(entities) + 2,
                                  j * 2 - width + 1,
                                  i * 2 - height)
                lamps[(i, j)] = lamp
                add_bidirectional_connection(combinator, lamp)
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

def convert_image_to_blueprint(image, show_intermediates, clusters):
    width, height = image.size
    flat_image = np.asarray(image, dtype=np.float32)
    if flat_image.shape[2] == 4:
        # ignore alpha channel
        flat_image = flat_image[:, :, :3]
    elif flat_image.shape[2] != 3:
        raise RuntimeError("Only works on RGB images.  Color depth: %d" %
                           flat_image.shape[2])
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


def convert_blueprint_to_preview(blueprint):
    """
    Converts one of the blueprints created above back to an image.

    Useful for displaying previews and for verifying that the
    blueprint was sensible.
    """
    entities = decompress_blueprint(blueprint)["blueprint"]["entities"]
    horizon = deque()

    entity_map = {}
    entity_colors = {}
    for e in entities:
        entity_map[e["entity_number"]] = e
        if e["name"] == "constant-combinator":
            horizon.append(e["entity_number"])
            color = e["control_behavior"]["filters"][0]["signal"]["name"]
            entity_colors[e["entity_number"]] = color

    while len(horizon) > 0:
        e = horizon.popleft()
        color = entity_colors[e]
        for n in entity_map[e]["connections"]["1"]["green"]:
            if n["entity_id"] not in entity_colors:
                horizon.append(n["entity_id"])
                entity_colors[n["entity_id"]] = color
    
    lamps = [e for e in entities if e["name"] == "small-lamp"]
    min_x = min(e["position"]["x"] for e in lamps)
    min_y = min(e["position"]["y"] for e in lamps)
    max_x = max(e["position"]["x"] for e in lamps)
    max_y = max(e["position"]["y"] for e in lamps)

    width = max_x - min_x + 6
    height = max_y - min_y + 6

    image = np.zeros((height, width, 3), dtype=np.int8)
    for lamp in lamps:
        x = lamp["position"]["x"] - min_x
        y = lamp["position"]["y"] - min_y
        color = entity_colors[lamp["entity_number"]]
        color = BASE_COLORS[color]
        image[y+2, x+2, :] = color
        image[y+3, x+2, :] = color
        image[y+2, x+3, :] = color
        image[y+3, x+3, :] = color

    return Image.fromarray(image, "RGB")

def resize_image(image, shape=None, lamps=None, default=False):
    print("Original image size: %s" % str(image.size))
    if shape:
        if lamps or default:
            raise RuntimeError("Can only specify one resize method")
        new_width, new_height = shape
    elif default:
        if lamps:
            raise RuntimeError("Can only specify one resize method")
        width, height = image.size
        if width > height:
            new_width = 90
            new_height = int(height / width * new_width)
        elif height > width:
            new_height = 90
            new_width = int(width / height * new_height)
        else:
            new_height = 90
            new_width = 90
    elif lamps:
        width, height = image.size
        max_d = max(width, height)
        min_d = min(width, height)
        scaled_min = (lamps / (max_d / min_d)) ** 0.5
        scaled_max = int(scaled_min * max_d / min_d)
        scaled_min = int(scaled_min)
        if width > height:
            new_width = scaled_max
            new_height = scaled_min
        else:
            new_height = scaled_max
            new_width = scaled_min
    else:
        raise RuntimeError("No resize method specified")

    new_width = max(1, new_width)
    new_height = max(1, new_height)
    print("Resizing to %d, %d" % (new_width, new_height))
    image = image.resize((new_width, new_height))
    return image
            
if __name__ == '__main__':
    path = sys.argv[1]

    image = Image.open(path)
    if len(sys.argv) > 2:
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        shape = (width, height)
        image = resize_image(image, shape=shape)

    show_intermediates = False
    if show_intermediates:
        image.show()

    bp = convert_image_to_blueprint(image, show_intermediates, 7)
    print
    print("BLUEPRINT")
    print(bp)

    preview = convert_blueprint_to_preview(bp)
    preview.show()
