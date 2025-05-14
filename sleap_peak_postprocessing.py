import tensorflow as tf
from typing import Optional, Tuple

def integral_regression(
    cms: tf.Tensor, xv: tf.Tensor, yv: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute regression by integrating over the confidence maps on a grid.

    Args:
        cms: Confidence maps with shape (samples, height, width, channels).
        xv: X grid vector tf.float32 of grid coordinates to sample.
        yv: Y grid vector tf.float32 of grid coordinates to sample.

    Returns:
        A tuple of (x_hat, y_hat) with the regressed x- and y-coordinates for each
        channel of the confidence maps.

        x_hat and y_hat are of shape (samples, channels)
    """
    # Compute normalizing factor.
    z = tf.reduce_sum(cms, axis=[1, 2])

    # Regress to expectation.
    x_hat = tf.reduce_sum(tf.reshape(xv, [1, 1, -1, 1]) * cms, axis=[1, 2]) / z
    y_hat = tf.reduce_sum(tf.reshape(yv, [1, -1, 1, 1]) * cms, axis=[1, 2]) / z

    return x_hat, y_hat

def make_centered_bboxes(
    centroids: tf.Tensor, box_height: int, box_width: int
) -> tf.Tensor:
    """Generate bounding boxes centered on a set of centroid coordinates.

    Args:
        centroids: A tensor of shape (n_centroids, 2) and dtype tf.float32, where the
            last axis corresponds to the (x, y) coordinates of each centroid.
        box_height: Scalar integer indicating the height of the bounding boxes.
        box_width: Scalar integer indicating the width of the bounding boxes.

    Returns:
        Tensor of shape (n_centroids, 4) and dtype tf.float32, where the last axis
        corresponds to (y1, x1, y2, x2) coordinates of the bounding boxes in absolute
        image coordinates.

    Notes:
        The bounding box coordinates are calculated such that the centroid coordinates
        map onto the center of the pixel. For example:

        For a single row image of shape (1, 4) with values: `[[a, b, c, d]]`, the x
        coordinates can be visualized in the diagram below:
                 _______________________
                |  a  |  b  |  c  |  d  |
                |  |  |  |  |  |  |  |  |
              -0.5 | 0.5 | 1.5 | 2.5 | 3.5
                   0     1     2     3

        To get a (1, 3) patch centered at c, the centroid would be at (x, y) = (2, 0)
        with box height of 1 and box width of 3, to yield `[[b, c, d]]`.

        For even sized bounding boxes, e.g., to get the center 2 elements, the centroid
        would be at (x, y) = (1.5, 0) with box width of 2, to yield `[[b, c]]`.
    """
    delta = (
        tf.convert_to_tensor(
            [[-box_height + 1, -box_width + 1, box_height - 1, box_width - 1]],
            tf.float32,
        )
        * 0.5
    )
    bboxes = tf.gather(centroids, [1, 0, 1, 0], axis=-1) + delta
    return bboxes

def normalize_bboxes(
    bboxes: tf.Tensor, image_height: int, image_width: int
) -> tf.Tensor:
    """Normalize bounding box coordinates to the range [0, 1].

    This is useful for transforming points for TensorFlow operations that require
    normalized image coordinates.

    Args:
        bboxes: Tensor of shape (n_bboxes, 4) and dtype tf.float32, where the last axis
            corresponds to (y1, x1, y2, x2) coordinates of the bounding boxes.
        image_height: Scalar integer indicating the height of the image.
        image_width: Scalar integer indicating the width of the image.

    Returns:
        Tensor of the normalized points of the same shape as `bboxes`.

        The normalization applied to each point is `x / (image_width - 1)` and
        `y / (image_width - 1)`.

    See also: unnormalize_bboxes
    """
    # Compute normalizing factor of shape (1, 4).
    factor = (
        tf.convert_to_tensor(
            [[image_height, image_width, image_height, image_width]], tf.float32
        )
        - 1
    )

    # Normalize and return.
    normalized_bboxes = bboxes / factor
    return normalized_bboxes

def crop_bboxes(
    images: tf.Tensor, bboxes: tf.Tensor, sample_inds: tf.Tensor
) -> tf.Tensor:
    """Crop bounding boxes from a batch of images.

    This method serves as a convenience method for specifying the arguments of
    `tf.image.crop_and_resize`.

    Args:
        images: Tensor of shape (samples, height, width, channels) of a batch of images.
        bboxes: Tensor of shape (n_bboxes, 4) and dtype tf.float32, where the last axis
            corresponds to unnormalized (y1, x1, y2, x2) coordinates of the bounding
            boxes. This can be generated from centroids using `make_centered_bboxes`.
        sample_inds: Tensor of shape (n_bboxes,) specifying which samples each bounding
            box should be cropped from.

    Returns:
        A tensor of shape (n_bboxes, crop_height, crop_width, channels) of the same
        dtype as the input image. The crop size is inferred from the bounding box
        coordinates.

    Notes:
        This function expects bounding boxes with coordinates at the centers of the
        pixels in the box limits. Technically, the box will span (x1 - 0.5, x2 + 0.5)
        and (y1 - 0.5, y2 + 0.5).

        For example, a 3x3 patch centered at (1, 1) would be specified by
        (y1, x1, y2, x2) = (0, 0, 2, 2). This would be exactly equivalent to indexing
        the image with `image[0:3, 0:3]`.

    See also: `make_centered_bboxes`
    """
    # Compute bounding box size to use for crops.
    y1x1 = tf.gather_nd(bboxes, [[0, 0], [0, 1]])
    y2x2 = tf.gather_nd(bboxes, [[0, 2], [0, 3]])
    box_size = tf.cast(tf.math.round((y2x2 - y1x1) + 1), tf.int32)  # (height, width)

    # Normalize bounding boxes.
    image_height = tf.shape(images)[1]
    image_width = tf.shape(images)[2]
    normalized_bboxes = normalize_bboxes(
        bboxes, image_height=image_height, image_width=image_width
    )

    # Crop.
    crops = tf.image.crop_and_resize(
        images,
        boxes=normalized_bboxes,
        box_indices=tf.cast(sample_inds, tf.int32),
        crop_size=box_size,
        method="bilinear",
    )

    # Cast back to original dtype and return.
    crops = tf.cast(crops, images.dtype)
    return crops

def find_local_peaks_rough(
    cms: tf.Tensor, threshold: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Find local maxima via non-maximum suppresion.

    Args:
        cms: Tensor of shape (samples, height, width, channels).
        threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
            with values below this threshold will not be returned.

    Returns:
        A tuple of (peak_points, peak_vals, peak_sample_inds, peak_channel_inds).
        peak_points: float32 tensor of shape (n_peaks, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (n_peaks,) containing the values at the peak
        points.

        peak_sample_inds: int32 tensor of shape (n_peaks,) containing the indices of the
        sample each peak belongs to.

        peak_channel_inds: int32 tensor of shape (n_peaks,) containing the indices of
        the channel each peak belongs to.
    """
    # Build custom local NMS kernel.
    kernel = tf.reshape(
        tf.constant([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=tf.float32), (3, 3, 1)
    )

    # Reshape to have singleton channels.
    height = tf.shape(cms)[1]
    width = tf.shape(cms)[2]
    channels = tf.shape(cms)[3]
    flat_img = tf.reshape(tf.transpose(cms, [0, 3, 1, 2]), [-1, height, width, 1])

    # Perform dilation filtering to find local maxima per channel and reshape back.
    max_img = tf.nn.dilation2d(
        flat_img, kernel, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]
    )
    max_img = tf.transpose(
        tf.reshape(max_img, [-1, channels, height, width]), [0, 2, 3, 1]
    )

    # Filter for maxima and threshold.
    argmax_and_thresh_img = (cms > max_img) & (cms > threshold)

    # Convert to subscripts.
    peak_subs = tf.where(argmax_and_thresh_img)

    # Get peak values.
    peak_vals = tf.gather_nd(cms, peak_subs)

    # Convert to points format.
    peak_points = tf.cast(tf.gather(peak_subs, [2, 1], axis=1), tf.float32)

    # Pull out indexing vectors.
    peak_sample_inds = tf.cast(tf.gather(peak_subs, 0, axis=1), tf.int32)
    peak_channel_inds = tf.cast(tf.gather(peak_subs, 3, axis=1), tf.int32)

    return peak_points, peak_vals, peak_sample_inds, peak_channel_inds

def find_local_peaks(
    cms: tf.Tensor,
    threshold: float = 0.2,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    A tailed version of find_local_peaks in SLEAP.
    Find local peaks with optional refinement.

    Args:
        cms: Confidence maps. Tensor of shape (samples, height, width, channels).
        threshold: Minimum confidence threshold. Peaks with values below this will
            ignored.
    Returns:
        A tuple of (peak_points, peak_vals, peak_sample_inds, peak_channel_inds).

        peak_points: float32 tensor of shape (n_peaks, 2), where the last axis
        indicates peak locations in xy order.

        peak_vals: float32 tensor of shape (n_peaks,) containing the values at the peak
        points.

        peak_sample_inds: int32 tensor of shape (n_peaks,) containing the indices of the
        sample each peak belongs to.

        peak_channel_inds: int32 tensor of shape (n_peaks,) containing the indices of
        the channel each peak belongs to.
    """
    # Find grid aligned peaks.
    (
        rough_peaks,
        peak_vals,
        peak_sample_inds,
        peak_channel_inds,
    ) = find_local_peaks_rough(cms, threshold=threshold)

    if tf.shape(rough_peaks)[0] == 0:
        # No peaks found. Return empty tensors.
        return tf.zeros([0, 2], tf.float32), tf.zeros([0], tf.float32), tf.zeros([0], tf.int32), tf.zeros([0], tf.int32)

    crop_size = 5 # Default size for local gaussian distribution. (default sigma = 2.5 in sleap)

    # Make bounding boxes for cropping around peaks.
    bboxes = make_centered_bboxes(
        rough_peaks, box_height=crop_size, box_width=crop_size
    )

    # Reshape to (samples * channels, height, width, 1).
    n_samples = tf.shape(cms)[0]
    n_channels = tf.shape(cms)[3]
    cms = tf.reshape(
        tf.transpose(cms, [0, 3, 1, 2]),
        [n_samples * n_channels, tf.shape(cms)[1], tf.shape(cms)[2], 1],
    )
    box_sample_inds = (peak_sample_inds * n_channels) + peak_channel_inds

    # Crop patch around each grid-aligned peak.
    cm_crops = crop_bboxes(cms, bboxes, sample_inds=box_sample_inds)

    # Compute offsets via integral regression on a local patch.
    gv = tf.cast(tf.range(crop_size), tf.float32) - ((crop_size - 1) / 2)
    dx_hat, dy_hat = integral_regression(cm_crops, xv=gv, yv=gv)
    offsets = tf.concat([dx_hat, dy_hat], axis=1)

    # Apply offsets.
    refined_peaks = rough_peaks + offsets

    return refined_peaks, peak_vals, peak_sample_inds, peak_channel_inds