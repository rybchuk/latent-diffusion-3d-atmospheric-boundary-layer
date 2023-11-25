# Based off of https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models
import numpy as np
import torch

def random_bbox_3d(img_shape=(32, 32, 32), max_bbox_shape=(16, 16, 16), max_bbox_delta=12, min_margin=2):
    '''
    Same as random_bbox, but for volumes
    
    Returns:
        tuple[int]: The generated box, (top, left, front, h, w, d)
    '''
    # Modify inputs if they aren't tuples already
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin, min_margin)

    # Mask parameters
    img_h, img_w, img_d = img_shape[:3]
    max_mask_h, max_mask_w, max_mask_d = max_bbox_shape
    max_delta_h, max_delta_w, max_delta_d = max_bbox_delta
    margin_h, margin_w, margin_d = min_margin

    # Check mask parameters
    if max_mask_h > img_h or max_mask_w > img_w or max_mask_d > img_d:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w
            or max_delta_d // 2 * 2 >= max_mask_d):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if (img_h - max_mask_h < 2 * margin_h 
            or img_w - max_mask_w < 2 * margin_w
            or img_d - max_mask_d < 2 * margin_d):
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    ## Create mask
    # get the max value of (top, left, front)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    max_front = img_d - margin_d - max_mask_d
    # randomly select a (top, left, front)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    front = np.random.randint(margin_d, max_front)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    delta_front = np.random.randint(0, max_delta_d // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    front = front + delta_front
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    d = max_mask_d - delta_front
    return (top, left, front, h, w, d)

def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, [d], 1). '0' indicates the
    hole and '1' indicates the observed regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, [front], height, width, [depth])
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, [d], 1).
    """

    if len(img_shape) == 2:  # Plane data
        height, width = img_shape[:2]

        mask = np.ones((1, 1, height, width), dtype=dtype)
        mask[:, :, bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3]] = 0
    elif len(img_shape) == 3:  # Volume data
        height, width, depth = img_shape[:3]

        mask = np.ones((1, 1, height, width, depth), dtype=dtype)
        mask[:, :, bbox[0]:bbox[0] + bbox[3], bbox[1]:bbox[1] + bbox[4], bbox[2]:bbox[2] + bbox[5]] = 0

    return torch.from_numpy(mask)

def get_stripe_mask(img_shape, x0_stripe=4, dx_stripe=4, n_stripes=7, axis=0, dtype='uint8'):
    '''
    Create a mask for volumes with multiple stripes of visibility

    The orientation of the stripes changes with `axis`
    '''

    ## Check inputs
    if axis > 3: raise ValueError("Value of axis must be < 3!")
    if x0_stripe + n_stripes*dx_stripe > img_shape[axis]:
        raise ValueError("Requested mask inputs exceed image shape!")

    ## Initialize mask
    height, width, depth = img_shape[:3]
    mask = np.zeros((1, 1, height, width, depth), dtype=dtype)

    ## Populate mask
    for i in range(n_stripes):
        xloc = x0_stripe + i*dx_stripe
        if axis == 0:
            mask[0,0,xloc,:,:] = 1
        elif axis == 1:
            mask[0,0,:,xloc,:] = 1
        elif axis == 2:
            mask[0,0,:,:,xloc] = 1

    return torch.from_numpy(mask)

def get_custom_mask(img_shape, mask_dir, dtype='uint8'):
    '''
    Return a mask based on a user-specified numpy array
    '''

    ## Read the mask file
    mask = np.load(mask_dir)
    mask = mask.astype(dtype)

    ## Check that the mask dimensions are correct 
    height, width = img_shape[:2]
    assert mask.shape[2] == height
    assert mask.shape[3] == width

    if len(img_shape) == 3:  # Volume data
        depth = img_shape[2]
        assert mask.shape[4] == depth

    return torch.from_numpy(mask)