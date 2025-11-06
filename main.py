"""
Pyramidal Lucas-Kanade dense optical flow + frame interpolation

Usage:
    python lk_frame_interpolate.py frame1.jpg frame2.jpg out_interpolated.jpg

Notes:
- Works reasonably for small-to-moderate motion. Use more pyramid levels for larger motion.
- Parameters can be tuned for quality vs speed.
"""


from PIL import Image
import numpy as np


def rgb2gray(im):
    arr = np.asarray(im, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    return arr


def gaussian_kernel(k=5, sigma=1.0):
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def convolve2d(img, kernel):
    # simple 2D convolution, kernel assumed small
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    img_p = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(img)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = img_p[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * kernel)
    return out


def downsample(img):
    # simple blur then decimate by 2
    k = gaussian_kernel(5, 1.0)
    blurred = convolve2d(img, k)
    return blurred[::2, ::2]


def upsample(img, out_shape):
    # simple bilinear upsampling to out_shape (H, W)
    H, W = out_shape
    src_h, src_w = img.shape
    scale_y = src_h / H
    scale_x = src_w / W
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    src_y = (yy * scale_y)
    src_x = (xx * scale_x)
    return bilinear_interpolation(img, src_x, src_y)


def bilinear_interpolation(img, x, y):
    """
    img: H x W
    x, y: arrays of coordinates (float), shape matches output shape
    returns sampled image at (y, x) using bilinear interpolation
    Note: x = horizontal coordinate, y = vertical
    """
    H, W = img.shape
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def color_bilinear_interpolation(img, x, y):
    # img: H x W x C
    if img.ndim == 2:
        return bilinear_interpolation(img, x, y)
    H, W, C = img.shape
    out = np.zeros(x.shape + (C,), dtype=np.float32)
    for c in range(C):
        out[..., c] = bilinear_interpolation(img[..., c], x, y)
    return out


def compute_gradients(I1, I2):
    # central differences for spatial gradients, forward difference for temporal
    Ix = 0.5 * (np.roll(I1, -1, axis=1) - np.roll(I1, 1, axis=1) +
                np.roll(I2, -1, axis=1) - np.roll(I2, 1, axis=1)) * 0.5
    Iy = 0.5 * (np.roll(I1, -1, axis=0) - np.roll(I1, 1, axis=0) +
                np.roll(I2, -1, axis=0) - np.roll(I2, 1, axis=0)) * 0.5
    It = I2 - I1
    return Ix, Iy, It


def pyramidal_lucas_kanade(I1, I2, num_levels=3, win_size=5, max_iter=3, epsilon=1e-4):
    """
    Compute dense flow from I1 -> I2 using pyramidal Lucas-Kanade.
    I1, I2: grayscale images (float32, range 0..1)
    num_levels: pyramid levels (coarsest level is level 0)
    win_size: window size for local least squares (odd)
    max_iter: iterations per level for refinement (warping iterations)
    Returns:
        u, v arrays of flow (same shape as I1)
    """
    # build pyramids (level 0 = coarsest)
    pyr1 = [I1]
    pyr2 = [I2]
    for _ in range(1, num_levels):
        pyr1.insert(0, downsample(pyr1[0]))
        pyr2.insert(0, downsample(pyr2[0]))

    # initialize flow at coarsest scale
    Hc, Wc = pyr1[0].shape
    u = np.zeros((Hc, Wc), dtype=np.float32)
    v = np.zeros((Hc, Wc), dtype=np.float32)

    half = win_size // 2
    wkernel = gaussian_kernel(win_size, sigma=win_size/3.0)

    for lvl in range(num_levels):
        I1l = pyr1[lvl]
        I2l = pyr2[lvl]
        H, W = I1l.shape

        # if not first level (i.e. finer), upscale flow from previous level
        if lvl != 0:
            # upsample u,v to current level
            u = upsample(u, (H, W)) * 2.0
            v = upsample(v, (H, W)) * 2.0

        # iterative refinement with warping
        for it in range(max_iter):
            # warp I2 toward I1 using current flow (so we linearize around current estimate)
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            x_warp = xx + u
            y_warp = yy + v
            I2w = bilinear_interpolation(I2l, x_warp, y_warp)

            # compute gradients on the warped images
            Ix, Iy, It = compute_gradients(I1l, I2w)

            # per-pixel solve for flow increments du,dv within window using weighted least squares
            du = np.zeros_like(u)
            dv = np.zeros_like(v)

            # precompute products
            Ix2 = Ix * Ix
            Iy2 = Iy * Iy
            Ixy = Ix * Iy
            Ixt = Ix * It
            Iyt = Iy * It

            # convolve with window (weighted sums)
            # simple implementation of local sums via convolution with kernel
            S_Ix2 = convolve2d(Ix2, wkernel)
            S_Iy2 = convolve2d(Iy2, wkernel)
            S_Ixy = convolve2d(Ixy, wkernel)
            S_Ixt = convolve2d(Ixt, wkernel)
            S_Iyt = convolve2d(Iyt, wkernel)

            # Solve 2x2 system per pixel:
            # [S_Ix2  S_Ixy] [du] = -[S_Ixt]
            # [S_Ixy  S_Iy2] [dv]   [S_Iyt]
            denom = (S_Ix2 * S_Iy2 - S_Ixy * S_Ixy)
            # regularize denom
            denom_reg = denom + 1e-6  # to avoid division by zero

            du = (-S_Iy2 * S_Ixt + S_Ixy * S_Iyt) / denom_reg
            dv = (S_Ixy * S_Ixt - S_Ix2 * S_Iyt) / denom_reg

            # update flow
            u += du
            v += dv

            # small-check for convergence (optional)
            mean_update = (np.mean(np.abs(du)) + np.mean(np.abs(dv))) * 0.5
            if mean_update < epsilon:
                break

    # final flow at full resolution (if pyramid built with full-level last)
    # if the pyramid's last level is the original size then we are good
    # else, ensure u,v match original size by upsampling if needed
    if u.shape != I1.shape:
        u = upsample(u, I1.shape)
        v = upsample(v, I1.shape)

    return u, v


def warp_color(img, u, v):
    """
    Warp color image img (H x W x C) by flow u,v where the pixel at (y,x) comes from (y - v, x - u)
    Here we use forward mapping by sampling from source at (x - u, y - v)
    We'll implement backward sampling:
        dest(y,x) = src(y - v(y,x)*t, x - u(y,x)*t)
    For our usage, we pass target coordinates already computed.
    """
    H, W = img.shape[:2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # sample source at x_src = xx - u, y_src = yy - v
    x_src = xx - u
    y_src = yy - v
    return color_bilinear_interpolation(img, x_src, y_src)


def interpolate_frame(img1, img2, u, v, t=0.5):
    """
    Produce interpolated frame at time t in [0,1] where t=0.5 is halfway.
    We warp both images toward the t position.
    For simplicity, warp img1 forward by t*(u,v) and img2 backward by (1-t)*(u,v)
    Here u,v are flow from img1->img2.
    """
    # warp forward img1 by +t*u, +t*v (move pixels towards their positions in img2)
    H, W = u.shape
    # compute maps for sampling: sample source at (x - t*u, y - t*v)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x1_src = xx - t * u
    y1_src = yy - t * v
    img1_w = color_bilinear_interpolation(img1, x1_src, y1_src)

    # warp backward img2 by -(1-t)*u -> sample img2 at (x + (1-t)*u, y + (1-t)*v)
    x2_src = xx + (1.0 - t) * u
    y2_src = yy + (1.0 - t) * v
    img2_w = color_bilinear_interpolation(img2, x2_src, y2_src)

    # simple blend
    out = (1.0 - t) * img1_w + t * img2_w
    out = np.clip(out, 0.0, 1.0)
    return out


def main(path1, path2, outpath):
    # Load images
    im1 = Image.open(path1).convert('RGB')
    im2 = Image.open(path2).convert('RGB')

    # optionally resize to smaller for speed (uncomment if needed)
    # target_size = (640, 360)
    # im1 = im1.resize(target_size, Image.BILINEAR)
    # im2 = im2.resize(target_size, Image.BILINEAR)

    # ensure same size
    if im1.size != im2.size:
        print("Resizing second image to match first.")
        im2 = im2.resize(im1.size, Image.BILINEAR)

    # convert to numpy arrays
    img1 = np.asarray(im1, dtype=np.float32) / 255.0
    img2 = np.asarray(im2, dtype=np.float32) / 255.0

    # grayscale
    g1 = rgb2gray(im1)
    g2 = rgb2gray(im2)

    # parameters (tweak these)
    num_levels = 8        # pyramid levels (increase for large motion)
    win_size = 7          # window for LK
    max_iter = 3          # iterations per pyramid level

    print("Computing flow with pyramidal Lucas-Kanade...")
    u, v = pyramidal_lucas_kanade(g1, g2, num_levels=num_levels, win_size=win_size, max_iter=max_iter)

    print("Warping and interpolating halfway frame...")
    interp = interpolate_frame(img1, img2, u, v, t=0.5)

    out_img = Image.fromarray(np.uint8(interp * 255.0))
    out_img.save(outpath)
    print("Saved interpolated frame to", outpath)


main("frame1.png", "frame2.png", "frame_gen1.png")
