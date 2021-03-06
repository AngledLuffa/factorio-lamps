import base64
import json
import warnings
import sys
import zlib

from collections import deque, namedtuple, OrderedDict

import skimage.color
import networkx as nx
import numpy as np
from scipy.cluster.vq import kmeans2
from PIL import Image
from PIL.ExifTags import TAGS as EXIF_TAGS

SHOW_INTERMEDIATES = False
SHOW_BLUEPRINT = False
SHOW_PREVIEW = True

ColorEntry = namedtuple('ColorEntry',
                        ['name', 'RGB', 'LAB'])

def BuildColorInfo(colors):
    processed_colors = []
    for color in colors.keys():
        rgb = colors[color]
        # LAB colors are for more accurate color assignment
        # https://en.wikipedia.org/wiki/Color_difference#CIEDE2000
        lab = skimage.color.rgb2lab(rgb.reshape((1, 1, 3)) / 256)
        processed_colors.append(ColorEntry(name=color,
                                           RGB=rgb,
                                           LAB=lab.reshape((3))))
    return processed_colors

BASE_COLORS = BuildColorInfo({
    "signal-red": np.array((255, 0, 0)),
    "signal-green": np.array((0, 255, 0)),
    "signal-blue": np.array((0, 0, 255)),
    "signal-yellow": np.array((255, 255, 0)),
    "signal-pink": np.array((255, 0, 255)),
    "signal-cyan": np.array((0, 255, 255)),
    "signal-white": np.array((255, 255, 255)),
    "signal-black": np.array((0, 0, 0)),
})

EXPANDED_LAMP_COLORS = BuildColorInfo({
    "signal-white": np.array((255, 255, 255)),
    "signal-light-grey": np.array((228, 228, 228)),
    "signal-grey": np.array((136, 136, 136)),
    "signal-black": np.array((34, 34, 34)),
    "signal-pink": np.array((255, 167, 209)),
    "signal-red": np.array((229, 0, 0)),
    "signal-orange": np.array((229, 149, 0)),
    "signal-brown": np.array((160, 106, 66)),
    "signal-yellow": np.array((229, 217, 0)),
    "signal-light-green": np.array((148, 224, 68)),
    "signal-green": np.array((2, 190, 1)),
    "signal-cyan": np.array((0, 211, 221)),
    "signal-light-blue": np.array((0, 131, 199)),
    "signal-blue": np.array((0, 0, 234)),
    "signal-light-purple": np.array((207, 110, 228)),
    "signal-dark-purple": np.array((130, 0, 128)),
})

DECTORIO_LAMP_COLORS = BuildColorInfo({
    "signal-red": np.array((255, 40, 25)),
    "signal-orange": np.array((252, 112, 56)),
    "signal-tangerine": np.array((255, 147, 35)),
    "signal-yellow": np.array((255, 244, 68)),
    "signal-green": np.array((0, 242, 43)),
    "signal-cyan": np.array((2, 249, 255)),
    "signal-aqua": np.array((12, 170, 252)),
    "signal-blue": np.array((17, 89, 249)),
    "signal-purple": np.array((165, 96, 252)),
    "signal-pink": np.array((255, 107, 252)),
    "signal-maroon": np.array((127, 0, 0)),
    "signal-brown": np.array((153, 99, 35)),
    "signal-olive": np.array((127, 127, 2)),
    "signal-emerald": np.array((43, 137, 63)),
    "signal-teal": np.array((71, 153, 142)),
    "signal-navy": np.array((0, 0, 127)),
    "signal-violet": np.array((142, 30, 178)),
    "signal-black": np.array((56, 33, 142)),
    "signal-grey": np.array((204, 204, 204)),
    "signal-white": np.array((255, 255, 255)),
})

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

def min_cost_colors(centroids, colors):
    """
    Assign colors based on min cost flow from centroid to color.
    """
    K = len(centroids)
    assert K <= len(colors)

    # We build a mincost flow as follows:
    # source: K output
    # edges to each color: cost 0, flow 1
    # edges from each color to each centroid: cost L2, flow 1
    # edges from each centroid to sink: cost 0, flow 1
    # sink: K input

    centroid_names = ["C%d" % i for i in range(len(centroids))]
    rgb_centroids = [c.reshape((1, 1, 3)) / 256 for c in centroids]
    lab_centroids = [skimage.color.rgb2lab(rgb).reshape((3))
                     for rgb in rgb_centroids]

    G = nx.DiGraph()
    G.add_node('source', demand=-K)
    G.add_node('sink', demand=K)
    for c in centroid_names:
        G.add_node(c)
        G.add_edge('source', c, capacity=1, weight=0)
    for k in colors:
        G.add_node(k.name)
        for c in range(len(centroids)):
            distance = skimage.color.deltaE_ciede2000(k.LAB, lab_centroids[c])
            G.add_edge(centroid_names[c], k.name, capacity=1,
                       weight=int(distance))
        G.add_edge(k.name, 'sink', capacity=1, weight=0)
            
    flow = nx.algorithms.min_cost_flow(G)
    flow_colors = []
    for source in sorted(flow.keys()):
        if source == 'source':
            continue
        c = None
        for dest in flow[source]:
            if flow[source][dest] > 0:
                if c:
                    # TODO: can this happen if floating point result?
                    raise RuntimeError("Multiple colors mapped to same source")
                flow_colors.append(dest)
    return flow_colors
                
def build_combinator(entity_number, x, y, color, enabled):
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
                    "count": 1 if enabled else 0,
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

def convert_entities_to_blueprint(entities):
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

    blueprint["blueprint"]["entities"] = entities
    
    return blueprint

# All neighbors with -X or 0X and -Y within 10 spaces
# Greatly cuts down the number of combinators needed
# Ordered like this so the wires don't get excessively long
# -1,0 and 0,-1 are handled separately
POSSIBLE_NEIGHBORS = [(-1, -1), (-1, 1),
                      (-2, 0), (0, -2),
                      (-2, -1), (-1, -2), (-2, 1), (-1, 2),
                      (-3, 0), (0, -3), (-2, -2), (-2, 2), 
                      (-1, 3), (-3, 1), (-3, -1), (-1, -3),
                      (-3, 2), (-3, -2), (-4, 0), (0, -4),
                      (-4, 2), (-4, 1), (-5, 0), (-4, -1), (-4, -2), 
                      (-1, 4), (-2, 4), (-3, 4),
                      (-2, 3), (-3, 3), (-4, 3), 
                      (-2, -3), (-3, -3), (-4, -3),
                      (-1, -4), (-2, -4), (-3, -4), (0, -5)]
                      
def convert_to_blueprint(pixel_colors, width, height,
                         disable_black):
    entities = []
    lamps = {}
    lamp_groupings = {}
    grouping_lamps = {}

    for i in range(height):
        for j in range(width):
            entity_num = len(entities) + 1
            lamp = build_lamp(entity_num,
                              j * 2 - width + 1,
                              i * 2 - height)
            lamps[(i, j)] = lamp
            lamp_groupings[(i, j)] = entity_num
            grouping_lamps[entity_num] = [(i, j)]
            entities.append(lamp)

    def merge_groupings(x, y):
        if x == y:
            return False
        if len(grouping_lamps[y]) > len(grouping_lamps[x]):
            x, y = y, x
        for (i, j) in grouping_lamps[y]:
            lamp_groupings[(i, j)] = x
        grouping_lamps[x].extend(grouping_lamps[y])
        grouping_lamps[y] = []
        return True

    for i in range(height):
        for j in range(width-1):
            if pixel_colors[i][j] == pixel_colors[i][j+1]:
                if merge_groupings(lamp_groupings[(i, j)],
                                   lamp_groupings[(i, j+1)]):
                    add_bidirectional_connection(lamps[(i, j)],
                                                 lamps[(i, j+1)])
    for i in range(height-1):
        for j in range(width):
            if pixel_colors[i][j] == pixel_colors[i+1][j]:
                if merge_groupings(lamp_groupings[(i, j)],
                                   lamp_groupings[(i+1, j)]):
                    add_bidirectional_connection(lamps[(i, j)],
                                                 lamps[(i+1, j)])

    for i in range(height):
        for j in range(width):
            for ni, nj in POSSIBLE_NEIGHBORS:
                if i+ni < 0 or i+ni >= height:
                    continue
                if j+nj < 0 or j+nj >= width:
                    continue
                if pixel_colors[i][j] == pixel_colors[i+ni][j+nj]:
                    if merge_groupings(lamp_groupings[(i, j)],
                                       lamp_groupings[(i+ni, j+nj)]):
                        add_bidirectional_connection(lamps[(i, j)],
                                                     lamps[(i+ni, j+nj)])

    # add combinators for the colors
    for group, group_lamps in grouping_lamps.items():
        if len(group_lamps) == 0:
            continue

        i, j = group_lamps[0]
        color = pixel_colors[i][j]
        enabled = not (color == 'signal-black' and disable_black)
        combinator = build_combinator(len(entities) + 1,
                                      j * 2 - width,
                                      i * 2 - height,
                                      color, enabled)
        add_bidirectional_connection(lamps[(i, j)], combinator)
        entities.append(combinator)
    
    # add enough poles to cover the image
    pole_x_start = -width+3
    if pole_x_start % 2 == 1:
        pole_x_start = pole_x_start - 1
    pole_x = list(range(pole_x_start, width-2, 6))
    if len(pole_x) == 0:
        pole_x.append(pole_x_start)
    if pole_x[-1] < width - 3:
        pole_x.append(width - 3)
    pole_y_start = -height+3
    pole_y = list(range(pole_y_start, height-2, 8))
    if len(pole_y) == 0:
        pole_y.append(pole_y_start)
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

    return convert_entities_to_blueprint(entities)

def convert_image_to_array(image):
    image = np.asarray(image, dtype=np.float32)
    if len(image.shape) == 2:
        # BW image
        image = np.expand_dims(image, 2)
    elif len(image.shape) != 3:
        raise RuntimeError("Unknown matrix shape: %s" % str(image.shape))
    if image.shape[2] == 1:
        print("Converting BW by stacking 3 copies.  Efficiency be damned")
        image = np.tile(image, (1, 1, 3))
    if image.shape[2] == 4:
        # ignore alpha channel
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise RuntimeError("Only works on BW or RGB(a) images.  "
                           "Color depth: %d" % image.shape[2])
    return image

def nearest_colors(centroids, colors):
    label_to_colors = []
    for centroid in centroids:
        rgb = centroid.reshape((1, 1, 3)) / 256
        lab = skimage.color.rgb2lab(rgb).reshape((3))
        distances = [skimage.color.deltaE_ciede2000(color.LAB, lab)
                     for color in colors]
        label_to_colors.append(colors[np.argmin(distances)].name)
    return label_to_colors
            
def convert_image_to_blueprint_nearest(image, colors, disable_black):
    width, height = image.size
    flat_image = convert_image_to_array(image)
    flat_image = flat_image.reshape((width * height, 3))

    num_centroids = max(len(colors) * 2, 100)
    num_centroids = min(width * height, num_centroids)
    centroids, labels = kmeans2(flat_image, num_centroids,
                                iter=50, minit='points')

    # centroids will be a Kx3 array representing colors
    # labels will be which centroid for each pixel
    # so centroids[labels] will be the pixels mapped to their K colors
    flat_kmeans_image = centroids[labels]
    kmeans_image = flat_kmeans_image.reshape((height, width, 3))
    kmeans_image = np.array(kmeans_image, dtype=np.int8)

    new_image = Image.fromarray(kmeans_image, "RGB")

    label_to_colors = nearest_colors(centroids, colors)
    pixel_colors = np.array([label_to_colors[x] for x in labels])
    pixel_colors = pixel_colors.reshape((height, width))

    blueprint = convert_to_blueprint(pixel_colors, width, height,
                                     disable_black)
    return compress_blueprint(blueprint), new_image

def convert_image_to_blueprint_kmeans(image, colors, disable_black):
    width, height = image.size
    flat_image = convert_image_to_array(image)
    flat_image = flat_image.reshape((width * height, 3))

    num_centroids = min(len(colors), width * height)
    centroids, labels = kmeans2(flat_image, num_centroids,
                                iter=50, minit='points')

    # centroids will be a Kx3 array representing colors
    # labels will be which centroid for each pixel
    # so centroids[labels] will be the pixels mapped to their K colors
    flat_kmeans_image = centroids[labels]
    kmeans_image = flat_kmeans_image.reshape((height, width, 3))
    kmeans_image = np.array(kmeans_image, dtype=np.int8)

    new_image = Image.fromarray(kmeans_image, "RGB")

    label_to_colors = min_cost_colors(centroids, colors)
    pixel_colors = np.array([label_to_colors[x] for x in labels])
    pixel_colors = pixel_colors.reshape((height, width))

    blueprint = convert_to_blueprint(pixel_colors, width, height,
                                     disable_black)
    return compress_blueprint(blueprint), new_image


def convert_blueprint_to_preview(blueprint, colors):
    """
    Converts one of the blueprints created above back to an image.

    Useful for displaying previews and for verifying that the
    blueprint was sensible.
    """
    color_map = {}
    for color in colors:
        color_map[color.name] = color.RGB
    
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
        color = color_map[color]
        image[y+2, x+2, :] = color
        image[y+3, x+2, :] = color
        image[y+2, x+3, :] = color
        image[y+3, x+3, :] = color

    return Image.fromarray(image, "RGB")

def resize_image(image, shape=None, lamps=None, default=False):
    width, height = image.size
    if shape:
        if lamps or default:
            raise RuntimeError("Can only specify one resize method")
        new_width, new_height = shape
    elif default:
        if lamps:
            raise RuntimeError("Can only specify one resize method")
        if width < 90 and height < 90:
            new_width = width
            new_height = height
        elif width > height:
            new_width = 90
            new_height = int(height / width * new_width)
        elif height > width:
            new_height = 90
            new_width = int(width / height * new_height)
        else:
            new_height = 90
            new_width = 90
    elif lamps:
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
    if new_width == width and new_height == height:
        print("Original image size: %s.  Not resizing" %
              str(image.size))
    else:
        print("Original image size: %s.  Resizing to (%d, %d)" % 
              (str(image.size), new_width, new_height))
        image = image.resize((new_width, new_height))
    return image

# https://www.daveperrett.com/articles/2012/07/28/exif-orientation-handling-is-a-ghetto/
# rotation, flip
ORIENTATIONS = {
    1: (0, False),
    2: (0, True),
    3: (180, False),
    4: (180, True),
    5: (270, True),
    6: (270, False),
    7: (90, True),
    8: (90, False),
}

def get_exif_tag(image):
    """
    Returns an exif tag if possible, otherwise None.

    Some images don't have exif attached, and some image types don't
    even support exif.  We catch those errors and return None
    silently.
    """
    try:
        exif = image._getexif()
        if exif:
            for tag in image._getexif().keys():
                if EXIF_TAGS[tag] == 'Orientation':
                    return image._getexif()[tag]
    except AttributeError:
        # some images don't have exif
        pass
    return None

def open_rotated_image(path):
    """
    Opens an image from disk, applies exif rotation if relevant.
    """
    warnings.simplefilter('error', Image.DecompressionBombWarning)
    image = Image.open(path)
    orientation = get_exif_tag(image)
    if orientation:
        if orientation not in ORIENTATIONS:
            print("Unknown orientation %d" % orientation)
        else:
            rotation, flip = ORIENTATIONS.get(orientation)
            if rotation != 0:
                image = image.rotate(rotation, expand=True)
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def extract_blueprint_stats(bp):
    """
    Given a blueprint, extracts stats.

    Stats returned are how many of each item are in the blueprint.
    Items are alphabetized.
    """
    entities = decompress_blueprint(bp)["blueprint"]["entities"]
    names = {}
    for e in entities:
        names[e["name"]] = names.get(e["name"], 0) + 1
    stats = OrderedDict()
    for name, quantity in sorted(names.items()):
        stats[name] = quantity
    return stats

#COLORS = EXPANDED_LAMP_COLORS
COLORS = BASE_COLORS

if __name__ == '__main__':
    path = sys.argv[1]

    image = open_rotated_image(path)
    
    if len(sys.argv) > 2:
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        shape = (width, height)
        image = resize_image(image, shape=shape)

    if SHOW_INTERMEDIATES:
        image.show()

    bp, new_image = convert_image_to_blueprint_kmeans(image, COLORS, True)
    if SHOW_INTERMEDIATES:
        new_image.show()
    if SHOW_BLUEPRINT:
        print
        print("BLUEPRINT")
        print(bp)

    preview = convert_blueprint_to_preview(bp, COLORS)
    if SHOW_PREVIEW:
        preview.show()

    stats = extract_blueprint_stats(bp)
    entity_names = list(stats.keys())
    entity_names.sort()
    print()
    print("Entities used:")
    for name in entity_names:
        print("  %s: %d" % (name, stats[name]))

