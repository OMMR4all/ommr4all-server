from database.file_formats.pcgts import PcGts, Coords
import numpy as np
import logging
from PIL import Image
from typing import List, Optional

logger = logging.Logger(__name__)


class NoStaffLinesAvailable(Exception):
    pass


class NoStaffsAvailable(Exception):
    pass


def shape_to_rect(shape):
    assert(len(shape) == 2)
    return 0, 0, shape[0], shape[1]


def quad_as_rect(quad):
    if quad[0] != quad[2]: return False
    if quad[1] != quad[7]: return False
    if quad[4] != quad[6]: return False
    if quad[3] != quad[5]: return False
    return True


def quad_to_rect(quad):
    assert(len(quad) == 8)
    assert(quad_as_rect(quad))
    return quad[0], quad[1], quad[4], quad[3]


def rect_to_quad(rect):
    assert(len(rect) == 4)
    return rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1]


def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / w_div
    y_step = h / h_div
    y = rect[1]
    grid_vertex_matrix = []
    for _ in range(h_div + 1):
        grid_vertex_matrix.append([])
        x = rect[0]
        for _ in range(w_div + 1 ):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step

    return np.array(grid_vertex_matrix)


def transform(point, staffs: List[List[Coords]]):
    if len(staffs) == 0:
        raise NoStaffsAvailable

    x, y = point[0], point[1]
    top_staff_line_d = 10000000
    top_staff_line: Optional[Coords] = None
    bot_staff_line_d = 10000000
    bot_staff_line: Optional[Coords] = None
    for staff in staffs:
        for staff_line in staff:
            o_y = staff_line.interpolate_y(x)
            if o_y < y and y - o_y < top_staff_line_d:
                top_staff_line = staff_line
                top_staff_line_d = y - o_y

            if o_y > y and o_y - y < bot_staff_line_d:
                bot_staff_line = staff_line
                bot_staff_line_d = o_y - y

    if top_staff_line_d > 1000000:
        top_staff_line_d = bot_staff_line_d
        top_staff_line = bot_staff_line
    elif bot_staff_line_d > 1000000:
        bot_staff_line_d = top_staff_line_d
        bot_staff_line = top_staff_line

    if top_staff_line is None or bot_staff_line is None:
        raise NoStaffLinesAvailable

    top_offset = top_staff_line.center_y() - top_staff_line.interpolate_y(x)
    bot_offset = bot_staff_line.center_y() - bot_staff_line.interpolate_y(x)

    interp_y = np.interp(y, [top_staff_line.interpolate_y(x), bot_staff_line.interpolate_y(x)], [top_offset, bot_offset])

    #return x, np.rint(y - interp_y)
    return x, y - interp_y


def transform_grid(dst_grid, staves: List[List[Coords]], shape):
    src_grid = dst_grid.copy()

    for idx in np.ndindex(src_grid.shape[:2]):
        p = src_grid[idx]
        if shape[0] - 1 > p[0] > 0 and shape[1] - 1 > p[1] > 0:
            out = transform(p, staves)
            src_grid[idx][1] = out[1]

    return src_grid


def grid_to_mesh(src_grid, dst_grid):
    assert(src_grid.shape == dst_grid.shape)
    mesh = []
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
                        src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
                        src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
                        src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]
            dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
                        dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
                        dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
                        dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
            dst_rect = quad_to_rect(dst_quad)
            mesh.append([dst_rect, src_quad])
    return mesh


class Dewarper:
    def __init__(self, shape, staves: List[List[Coords]]):
        logger.info("Creating dewarper based on {} staves with shape {}".format(len(staves), shape))
        self.shape = shape
        self.dst_grid = griddify(shape_to_rect(self.shape), 10, 30)
        logger.debug("Transforming grid)")
        self.src_grid = transform_grid(self.dst_grid, staves, self.shape)
        logger.debug("Creating mesh")
        self.mesh = grid_to_mesh(self.src_grid, self.dst_grid)

    def dewarp(self, images, resamples: List[int] = None):
        if resamples is None:
            resamples = [0] * len(images)

        logger.debug("Transforming images based on mesh")
        out = [im.transform(im.size, Image.MESH, self.mesh, res) for im, res in zip(images, resamples)]
        logger.info("Finished")
        return out

    def inv_transform_point(self, p):
        p = np.asarray(p)
        for i, row in enumerate(self.dst_grid):
            for j, cell in enumerate(row):
                if (cell > p).all():
                    cell_origin = self.dst_grid[i - 1, j - 1]
                    rel = (p - cell_origin) / (cell - cell_origin)

                    target_cell_origin = self.src_grid[i - 1, j - 1]
                    target_cell_extend = self.src_grid[i, j] - target_cell_origin

                    return target_cell_origin + rel * target_cell_extend

        return p

    def inv_transform_points(self, ps):
        return np.array([self.inv_transform_point(p) for p in ps])

    def transform_point(self, p):
        p = np.asarray(p)
        for i, row in enumerate(self.src_grid):
            for j, cell in enumerate(row):
                if (cell > p).all():
                    cell_origin = self.src_grid[i - 1, j - 1]
                    rel = (p - cell_origin) / (cell - cell_origin)

                    target_cell_origin = self.dst_grid[i - 1, j - 1]
                    target_cell_extend = self.dst_grid[i, j] - target_cell_origin

                    return target_cell_origin + rel * target_cell_extend

        return p

    def transform_points(self, ps):
        return np.array([self.transform_point(p) for p in ps])


if __name__ == '__main__':
    from database import DatabaseBook
    from database.file_formats.pcgts import PageScaleReference
    import matplotlib.pyplot as plt
    page = DatabaseBook('Gothic_Test').pages()[0]
    binary = Image.open(page.file('binary_highres_preproc', create_if_not_existing=True).local_path())
    gray = Image.open(page.file('gray_highres_preproc').local_path())
    pcgts = PcGts.from_file(page.file('pcgts', create_if_not_existing=True))
    overlay = np.array(gray)

    points_to_transform = np.array([(100, 50), (200, 50), (300, 50), (400, 50), (600, 50), (800, 50), (100, 100), (200, 150), (300, 200)], dtype=int)

    # staffs.draw(overlay)
    images = [binary, gray, Image.fromarray(overlay)]
    f, ax = plt.subplots(2, len(images), True, True)
    for a, l in enumerate(images):
        l = np.array(l)
        for p in points_to_transform:
            l[p[1]-5:p[1]+5, p[0]-5:p[0]+5] = 255
        ax[0, a].imshow(l)

    dewarper = Dewarper(images[0].size, pcgts.page.all_staves_staff_line_coords(scale=PageScaleReference.HIGHRES))
    images = dewarper.dewarp(images)
    transformed_points = dewarper.transform_points(points_to_transform).astype(int)
    for a, l in enumerate(images):
        l = np.array(l)
        for p in transformed_points:
            l[p[1]-5:p[1]+5, p[0]-5:p[0]+5] = 255
        ax[1, a].imshow(l)

    plt.show()
