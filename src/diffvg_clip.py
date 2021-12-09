import torch
import pydiffvg
import clip
from PIL import Image
from clip_helper import MakeCutouts, ImageLoss
import diffvg_helper
import tqdm

import argparse

def main(args):
    from subprocess import call
    call(['rm', '-r', 'results/diffvg_clip'])

    # inputs
    # text_prompts = ["a painting of the sunset"]
    text_prompts_weights = [("a small girl in the grass plays", 1),]
    WIDTH, HEIGHT = 400, 200
    # WIDTH, HEIGHT = 224, 224 # cut size

    # setup diffvg
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    # setup clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    perceptor, preprocess = clip.load("ViT-B/32", device=device)

    # loss function
    cut_size = perceptor.visual.input_resolution
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
    img_feature_losses = []
    for text, weight in text_prompts_weights:
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            text_features = perceptor.encode_text(text)
        img_feature_losses.append(ImageLoss(embed=text_features, weight=weight))
    def loss_fcn(img):
        image_features = perceptor.encode_image(make_cutouts(img))
        return sum(img_feature_loss(image_features) for img_feature_loss in img_feature_losses)

    # # quick test
    # image = preprocess(Image.open("imgs/res_1.png")).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     image_features = perceptor.encode_image(make_cutouts(image))
    #     text_features = perceptor.encode_text(text)
    #     print(image.shape)
    #     print(image_features.shape)
    #     print(text_features.shape)
    #     print(loss_fcn(image))

    # Setup optimization
    shapes, shape_groups = diffvg_helper.initialize(args, WIDTH, HEIGHT)
    # Variables to optimize
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    for path in shapes:
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)
    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # initial
    with torch.no_grad():
        img = diffvg_helper.render(shapes, shape_groups, WIDTH, HEIGHT)
        pydiffvg.imwrite(img.cpu(), 'results/diffvg_clip/init.png', gamma=1)

    # Adam iterations.
    pbar = tqdm.trange(args.num_iter)
    for t in pbar:
        points_optim.zero_grad()
        width_optim.zero_grad()
        color_optim.zero_grad()

        # Forward pass: render the image.
        img = diffvg_helper.render(shapes, shape_groups, WIDTH, HEIGHT)
        # Save
        pydiffvg.imwrite(img.cpu(), 'results/diffvg_clip/iter_{:04}.png'.format(t), gamma=1)
        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('results/diffvg_clip/iter_{:04}.svg'.format(t), WIDTH, HEIGHT, shapes,
                              shape_groups)

        # loss
        img = diffvg_helper.convert_img(img)
        loss = loss_fcn(img)
        pbar.set_postfix({'loss': loss.item()})

        # Backpropagate & update.
        loss.backward()
        points_optim.step()
        width_optim.step()
        color_optim.step()

        # clamp
        for path in shapes:
            path.stroke_width.data.clamp_(1.0, args.max_stroke_width)
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

    # Render the final result.
    with torch.no_grad():
        img = diffvg_helper.render(shapes, shape_groups, WIDTH, HEIGHT)
        pydiffvg.imwrite(img.cpu(), 'results/diffvg_clip/final.png'.format(t), gamma=1)

    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", '-pattern_type', 'glob', "-i",
        "results/diffvg_clip/iter_*.png", "-vb", "20M",
        "results/diffvg_clip/out.mp4"])

    with open('results/diffvg_clip/args.txt', 'w') as f:
        f.write(str(args))

    import time
    fnamebase = '_'.join(list(zip(*text_prompts_weights))[0]).replace(' ', '-')
    foldername = time.strftime('%Y-%m-%d-%T')
    call(['cp', '-r', 'results/diffvg_clip', 'results3/{:}_{:}'.format(fnamebase, foldername)])
    print('results3/{:}'.format(fnamebase))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutn", type=int, default=32)
    parser.add_argument("--cut_pow", type=float, default=1.0)
    parser.add_argument("--num_paths", type=int, default=64)
    parser.add_argument("--max_stroke_width", type=int, default=25)
    parser.add_argument("--num_iter", type=int, default=500)
    args = parser.parse_args()
    main(args)
