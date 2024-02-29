# Pytorch Hessian Filter Layer

This is an implementation of an Hessian filter over a 2D image.

So the steps are:
- Smooth the image with a gaussian kernel
- Compute the hessian of the image.
- Compute eigenvalues of the hessian of the image.
- Filter the image

Sources of implementation:
- hessian filter in numpy: https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
- hessian filter: https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/filters/ridges.py#L172
- hessian matrix: https://github.com/scikit-image/scikit-image/blob/v0.22.0/skimage/feature/corner.py#L225
- reddit post: https://www.reddit.com/r/pytorch/comments/zahcsi/hessian_of_an_image/
- quora post: https://www.quora.com/What-are-the-ways-of-calculating-2-x-2-Hessian-matrix-for-2D-image-of-pixel-at-x-y-position
