import numpy as np

from mmpose.core.post_processing import transform_preds


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when normalize==0
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def pose_pck_accuracy(output, target, mask, thr=0.05, normalize=None):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)


def _transform_coords(coords, center, scale, output_size, use_udp=False):
    """
    coords: N K 2
    center: N 2
    scale: N 2
    output_size: list(2,)
    """
    scale = scale * 200.0
    if use_udp:
        target_scale = scale / (np.array(output_size) - 1.0)
    else:
        target_scale = scale / np.array(output_size)

    target_coords = coords.copy()
    target_coords = coords * target_scale[:, None, :] + center[:, None, :] - scale[:, None, :] * 0.5
    
    return target_coords


def instance_oks(output, target, scale, center, mask, normalize=None):
    """
    On time calculation of OKS per instance
    
    """
    N, K, H, W = output.shape

    sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    area = np.prod(scale * 200.0, axis=1, keepdims=True)
    
    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    pred = _transform_coords(pred, center, scale, [H, W])
    gt = _transform_coords(gt, center, scale, [H, W])
    
    # N, K
    mask = mask.copy()
    instance_mask = np.any(mask, axis=1)
    e = np.full((N, K), 1e6, dtype=np.float32)
    e[mask] = (((pred - gt) ** 2).sum(axis=2)/ vars[None, :] / (area + np.spacing(1)) / 2)[mask]
    e = np.exp(-e)
    oks = np.sum(e[instance_mask], axis=1) / np.sum(mask[instance_mask], axis=1)

    return oks, instance_mask

def instance_oks_by_coord(pred, gt, scale, center, mask, input_size=(192, 256)):
    """
    On time calculation of OKS per instance
    
    """
    N, K, C = pred.shape
    sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = (sigmas * 2) ** 2
    area = np.prod(scale * 200.0, axis=1, keepdims=True)
    size = np.array(input_size)
    pred *= size
    gt *= size
    pred = _transform_coords(pred, center, scale, input_size)
    gt = _transform_coords(gt, center, scale, input_size)

    mask = mask.copy()
    instance_mask = np.any(mask, axis=1)
    e = np.full((N, K), 1e6, dtype=np.float32)
    e[mask] = (((pred - gt) ** 2).sum(axis=2)/ vars[None, :] / (area + np.spacing(1)) / 2)[mask]
    e = np.exp(-e)
    oks = np.sum(e[instance_mask], axis=1) / np.sum(mask[instance_mask], axis=1)

    return oks, instance_mask

def instance_pck_accuracy(output, target, mask, thr=0.05, normalize=None, panalty_ratio=1.0):
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred, _ = _get_max_preds(output)
    gt, _ = _get_max_preds(target)
    
    _mask = mask.copy()
    _mask[np.where((normalize == 0).sum(1))[0], :] = False
    distances = np.full((N, K), 1e6, dtype=np.float32)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((pred - gt) / normalize[:, None, :])[_mask], axis=-1)
    
    instance_mask = np.any(_mask, axis=1)
    distances[distances > thr] *= panalty_ratio
    distances = np.exp(-distances)
    ick = distances.sum(axis=1)[instance_mask] / _mask.sum(axis=1)[instance_mask]
    
    return ick, instance_mask

def instance_pck_by_coord(pred, gt, mask, thr=0.05, panalty_ratio=1.0):
    N, K, C = pred.shape
    if K == 0:
        return None, 0, 0

    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, 
                       .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    vars = 2 * (sigmas) ** 2
    
    _mask = mask.copy()
    distances = np.full((N, K), 1e6, dtype=np.float32)
    distances[_mask] = (((pred - gt) ** 2).sum(axis=2) / vars[None, :])[_mask]
    
    instance_mask = np.any(_mask, axis=1)
    distances[distances > thr] *= panalty_ratio
    distances = np.exp(-distances)
    ick = distances.sum(axis=1)[instance_mask] / _mask.sum(axis=1)[instance_mask]
    
    return ick, instance_mask
    

def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    return acc, avg_acc, cnt

