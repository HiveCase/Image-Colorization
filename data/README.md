# Data

Put **colour** images here. The pipeline converts them to grayscale internally,
colorizes the grayscale, and compares against the original colour — this is how
we get a free ground-truth for the reference metrics.

## Layout

```
data/
├── raw/     # general colour photos (the main evaluation set)
└── ood/     # optional: genuinely out-of-domain material you curate
```

## Building a good stress dataset

The four stress conditions in the brief are best covered by *curating content*
in addition to the synthetic perturbations the code applies automatically:

| Stress condition        | What to include                                             |
|-------------------------|-------------------------------------------------------------|
| Out-of-domain           | X-rays, microscopy, thermal, satellite, sketches, paintings, screenshots, infrared |
| Complex lighting        | backlit portraits, sunsets, night scenes, mixed indoor light, strong shadows |
| Fine textures           | fur, hair, foliage, fabric weave, gravel, feathers, crowds  |
| Multi-object boundaries | street scenes, market stalls, plated food, cluttered desks  |

Aim for 20–50 images spread across these. The synthetic perturbations then
multiply each image into ~16 stress variants automatically.

For a rigorous quantitative baseline, a standard choice is a random subset of
the **ImageNet ctest10k** colorization split (used by the DDColor paper) plus
**COCO-Stuff** for multi-object scenes.
