# diffvg + CLIP

This was originally done as part of a class project for CS 7643 Fall 2021.

It is very similar to [CLIPDraw](https://arxiv.org/pdf/2106.14843.pdf) which I didn't learn about until I was tuning parameters for this and went to search for inspiration.

It is using [diffvg](https://github.com/BachiLi/diffvg) and [CLIP](https://github.com/openai/CLIP).

To run, install CLIP and diffvg, then:

```sh
python src/diffvg_clip 
  --num_paths 1024
  --max_stroke_width 25
  --num_iter 200
```

## diffvg examples

Also see `painterly_rendering.py` for the code to generate vector graphics given a raster image target.

PLEASE NOTE: THIS IS ALMOST COMPLETELY UNCHANGED FROM IT'S SOURCE EXAMPLE CODE IN THE DIFFVG REPOSITORY.  Find the original version [here](https://github.com/BachiLi/diffvg/blob/master/apps/painterly_rendering.py)

See `painterly.sh` for some command line parameters I ran it with, and use a diff-tool to see what changed from the original source.
