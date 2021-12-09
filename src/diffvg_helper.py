"""
This code was based on parts of
https://github.com/BachiLi/diffvg/blob/master/apps/painterly_rendering.py
with modifications.
"""

import torch
import pydiffvg
import random


def initialize(args, W, H):
    canvas_width, canvas_height = W, H
    num_paths = args.num_paths

    random.seed(1234)
    torch.manual_seed(1234)

    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.10
            p1 = (p0[0] + radius * (random.random() - 0.5),
                  p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5),
                  p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5),
                  p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        #points = torch.rand(3 * num_segments + 1, 2) * min(canvas_width, canvas_height)
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(1.0),
                             is_closed=False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=None,
                                         stroke_color=torch.tensor([
                                             random.random(),
                                             random.random(),
                                             random.random(),
                                             random.random()
                                         ]))
        shape_groups.append(path_group)

    return shapes, shape_groups


RENDER = pydiffvg.RenderFunction.apply


def render(shapes, shape_groups, canvas_width, canvas_height):
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    img = RENDER(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        0,  # seed
        None,
        *scene_args)

    # white background
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
        img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()) * (1 - img[:, :, 3:4])

    return img


def convert_img(img):
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
    return img
