#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches

# increase this number for larger tilings.
N_ITERATIONS = 4

num_tiles = 0

IDENTITY = [1, 0, 0, 0, 1, 0]

TILE_NAMES = ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi"]

COLOR_MAP_ORIG = {
    "Gamma": "rgb(255, 255, 255)",
    "Gamma1": "rgb(255, 255, 255)",
    "Gamma2": "rgb(255, 255, 255)",
    "Delta": "rgb(220, 220, 220)",
    "Theta": "rgb(255, 191, 191)",
    "Lambda": "rgb(255, 160, 122)",
    "Xi": "rgb(255, 242, 0)",
    "Pi": "rgb(135, 206, 250)",
    "Sigma": "rgb(245, 245, 220)",
    "Phi": "rgb(0,   255, 0)",
    "Psi": "rgb(0,   255, 255)"
}

COLOR_MAP_MYSTICS = {
    "Gamma": "rgb(196, 201, 169)",
    "Gamma1": "rgb(196, 201, 169)",
    "Gamma2": "rgb(156, 160, 116)",
    "Delta": "rgb(247, 252, 248)",
    "Theta": "rgb(247, 252, 248)",
    "Lambda": "rgb(247, 252, 248)",
    "Xi": "rgb(247, 252, 248)",
    "Pi": "rgb(247, 252, 248)",
    "Sigma": "rgb(247, 252, 248)",
    "Phi": "rgb(247, 252, 248)",
    "Psi": "rgb(247, 252, 248)"
}

COLOR_MAP = COLOR_MAP_ORIG


def is_rgb_tuple(value):
    return isinstance(value, tuple) and len(value) == 3 and all(0 <= v <= 1 for v in value)


def convert_rgb_string(rgb_string):
    if not isinstance(rgb_string, str):
        raise ValueError(f"Expected a string, but received {type(rgb_string)}")

    # Enlevez le "rgb(" du début et le ")" de la fin
    rgb_string = rgb_string[4:-1]

    # Divisez la chaîne en ses composants r, g, et b
    r, g, b = map(int, rgb_string.split(','))

    # Convertissez ces valeurs pour qu'elles soient comprises entre 0 et 1
    return r / 255, g / 255, b / 255


COLOR_MAP = {label: color if is_rgb_tuple(color) else convert_rgb_string(color) for label, color in COLOR_MAP.items()}


class pt:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xy = [x, y]


SPECTRE_POINTS = [
    pt(0, 0),
    pt(1.0, 0.0),
    pt(1.5, -np.sqrt(3) / 2),
    pt(1.5 + np.sqrt(3) / 2, 0.5 - np.sqrt(3) / 2),
    pt(1.5 + np.sqrt(3) / 2, 1.5 - np.sqrt(3) / 2),
    pt(2.5 + np.sqrt(3) / 2, 1.5 - np.sqrt(3) / 2),
    pt(3 + np.sqrt(3) / 2, 1.5),
    pt(3.0, 2.0),
    pt(3 - np.sqrt(3) / 2, 1.5),
    pt(2.5 - np.sqrt(3) / 2, 1.5 + np.sqrt(3) / 2),
    pt(1.5 - np.sqrt(3) / 2, 1.5 + np.sqrt(3) / 2),
    pt(0.5 - np.sqrt(3) / 2, 1.5 + np.sqrt(3) / 2),
    pt(-np.sqrt(3) / 2, 1.5),
    pt(0.0, 1.0)
]


def flatten(lst):
    return [item for sublist in lst for item in sublist]


SPECTRE_SHAPE = [(p.x, p.y) for p in SPECTRE_POINTS]


# Affine matrix multiply
def mul(A, B):
    result = [A[0] * B[0] + A[1] * B[3],
              A[0] * B[1] + A[1] * B[4],
              A[0] * B[2] + A[1] * B[5] + A[2],

              A[3] * B[0] + A[4] * B[3],
              A[3] * B[1] + A[4] * B[4],
              A[3] * B[2] + A[4] * B[5] + A[5]]
    return result


# Rotation matrix
def trot(ang):
    c = np.cos(ang)
    s = np.sin(ang)
    return [c, -s, 0, s, c, 0]


# Translation matrix
def ttrans(tx, ty):
    return [1, 0, tx, 0, 1, ty]


def transTo(p, q):
    return ttrans(q.x - p.x, q.y - p.y)


# Matrix * point
def transPt(M, P):
    return pt(M[0] * P.x + M[1] * P.y + M[2], M[3] * P.x + M[4] * P.y + M[5])


def drawPolygon(ax, T, f, s, w):
    """
    ax: matplotlib axis to draw on
    T: transformation matrix
    f: tile fill color
    s: tile stroke color
    w: tile stroke width
    """

    # Convertir la transformation 1D en une matrice 2D
    mtx = np.array([
        [T[0], T[1], T[2]],
        [T[3], T[4], T[5]],
        [0, 0, 1]
    ])
    trans = transforms.Affine2D(mtx) + ax.transData

    # Dessinez le polygone avec la transformation
    polygon = patches.Polygon(SPECTRE_SHAPE, closed=True, edgecolor=s, facecolor=f, linewidth=w, transform=trans)
    ax.add_patch(polygon)


class Tile:
    total_tiles = 0  # Attribut de classe pour suivre le nombre total de tuiles

    def __init__(self, pts, label):
        """
        pts: list of Tile coordinate points
        label: Tile type used for coloring
        """
        self.quad = [pts[3], pts[5], pts[7], pts[11]]
        self.label = label

    def draw(self, ax, tile_transformation=IDENTITY):

        global num_tiles
        num_tiles += 1

        mtx = np.array([
            [tile_transformation[0], tile_transformation[1], tile_transformation[2]],
            [tile_transformation[3], tile_transformation[4], tile_transformation[5]],
            [0, 0, 1]
        ])
        trans = transforms.Affine2D(mtx).scale(1, -1) + ax.transData

        polygon = patches.Polygon([(p.x, p.y) for p in SPECTRE_POINTS], closed=True, edgecolor="black",
                                  facecolor=COLOR_MAP[self.label], transform=trans)
        ax.add_patch(polygon)


class MetaTile:
    def __init__(self, geometries=[], quad=[]):
        """
        geometries: list of pairs of (Meta)Tiles and their transformations
        quad: MetaTile quad points
        """
        self.geometries = geometries
        self.quad = quad

    def draw(self, ax, metatile_transformation=IDENTITY, single_tile_label=None):
        """
        Recursively expand MetaTiles down to Tiles and draw those
        """
        for shape, shape_transformation in self.geometries:
            shape.draw(ax, mul(metatile_transformation, shape_transformation))


def draw_shape(shape_data):
    drawing, metatile_transformation, shape, shape_transformation = shape_data
    return shape.draw(drawing, mul(metatile_transformation, shape_transformation))


def buildSpectreBase():
    spectre_base_cluster = {label: Tile(SPECTRE_POINTS, label) for label in TILE_NAMES if label != "Gamma"}
    # special rule for Gamma
    mystic = MetaTile(
        [
            [Tile(SPECTRE_POINTS, "Gamma1"), IDENTITY],
            [Tile(SPECTRE_POINTS, "Gamma2"), mul(ttrans(SPECTRE_POINTS[8].x, SPECTRE_POINTS[8].y), trot(np.pi / 6))]
        ],
        [SPECTRE_POINTS[3], SPECTRE_POINTS[5], SPECTRE_POINTS[7], SPECTRE_POINTS[11]]
    )
    spectre_base_cluster["Gamma"] = mystic

    return spectre_base_cluster


def buildSupertiles(tileSystem):
    """
    iteratively build on current system of tiles
    tileSystem = current system of tiles, initially built with buildSpectreBase()
    """

    # First, use any of the nine-unit tiles in tileSystem to obtain
    # a list of transformation matrices for placing tiles within
    # supertiles.
    quad = tileSystem["Delta"].quad
    R = [-1, 0, 0, 0, 1, 0]

    """
    [rotation angle, starting quad point, target quad point]
    """
    transformation_rules = [
        [60, 3, 1], [0, 2, 0], [60, 3, 1], [60, 3, 1],
        [0, 2, 0], [60, 3, 1], [-120, 3, 3]
    ]

    transformations = [IDENTITY]
    total_angle = 0
    rotation = IDENTITY
    transformed_quad = list(quad)

    for _angle, _from, _to in transformation_rules:
        if (_angle != 0):
            total_angle += _angle
            rotation = trot(np.deg2rad(total_angle))
            transformed_quad = [transPt(rotation, quad_pt) for quad_pt in quad]

        ttt = transTo(
            transformed_quad[_to],
            transPt(transformations[-1], quad[_from])
        )
        transformations.append(mul(ttt, rotation))

    transformations = [mul(R, transformation) for transformation in transformations]

    # Now build the actual supertiles, labelling appropriately.
    super_rules = {
        "Gamma": ["Pi", "Delta", None, "Theta", "Sigma", "Xi", "Phi", "Gamma"],
        "Delta": ["Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
        "Theta": ["Psi", "Delta", "Pi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
        "Lambda": ["Psi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
        "Xi": ["Psi", "Delta", "Pi", "Phi", "Sigma", "Psi", "Phi", "Gamma"],
        "Pi": ["Psi", "Delta", "Xi", "Phi", "Sigma", "Psi", "Phi", "Gamma"],
        "Sigma": ["Xi", "Delta", "Xi", "Phi", "Sigma", "Pi", "Lambda", "Gamma"],
        "Phi": ["Psi", "Delta", "Psi", "Phi", "Sigma", "Pi", "Phi", "Gamma"],
        "Psi": ["Psi", "Delta", "Psi", "Phi", "Sigma", "Psi", "Phi", "Gamma"]
    }
    super_quad = [
        transPt(transformations[6], quad[2]),
        transPt(transformations[5], quad[1]),
        transPt(transformations[3], quad[2]),
        transPt(transformations[0], quad[1])
    ]

    return {
        label: MetaTile(
            [[tileSystem[substitution], transformation] for substitution, transformation in
             zip(substitutions, transformations) if substitution],
            super_quad
        ) for label, substitutions in super_rules.items()}


def is_within_boundary(tile, boundary):
    """
    Checks if a given tile lies within a specified boundary.
    For simplicity, this function just checks if the bounding box of the tile is within the boundary.
    A more advanced function might check every point of the tile, but this serves as a starting point.
    """
    min_x = min([point.x for point in tile.quad])
    max_x = max([point.x for point in tile.quad])
    min_y = min([point.y for point in tile.quad])
    max_y = max([point.y for point in tile.quad])

    boundary_min_x, boundary_max_x, boundary_min_y, boundary_max_y = boundary

    return min_x >= boundary_min_x and max_x <= boundary_max_x and min_y >= boundary_min_y and max_y <= boundary_max_y


def translate_point(point):
    return pt(point.x - 25, point.y - 25)


def translate_all_shapes(shapes, dx, dy):
    translation_matrix = ttrans(dx, dy)
    for shape in shapes.values():
        if isinstance(shape, MetaTile):
            for geometry, transformation in shape.geometries:
                if isinstance(geometry, Tile):
                    new_transformation = mul(translation_matrix, transformation)
                    geometry.transformation = new_transformation
        elif isinstance(shape, Tile):
            new_transformation = mul(translation_matrix, shape.transformation)
            shape.transformation = new_transformation


def main():
    global num_tiles
    num_tiles = 0

    user_x = float(input("Enter the x coordinate of the point : "))
    user_y = float(input("Enter the y coordinate of the point : "))

    shapes = buildSpectreBase()
    for _ in range(N_ITERATIONS):
        shapes = buildSupertiles(shapes)

    dx, dy = 25, 25
    translation = ttrans(dx, dy)

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.axis('equal')
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)

    for shape in shapes.values():
        shape.draw(ax, translation)
    ax.plot(user_x, user_y, 'ro')  # 'ro' signifie un point rouge

    ax.axvline(x=user_x, color='r', linestyle='--')
    ax.axhline(y=user_y, color='r', linestyle='--')

    plt.show()

    fig2, ax2 = plt.subplots(figsize=(15, 15))
    ax2.axis('equal')
    ax2.set_xlim(0, user_x)
    ax2.set_ylim(0, user_y)

    for shape in shapes.values():
        shape.draw(ax2)

    plt.show()


if __name__ == "__main__":
    main()
