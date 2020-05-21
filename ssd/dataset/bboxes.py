"""SSD bounding box utils."""
import torch


def convert_locations_to_boxes(
    locations: torch.Tensor,
    priors: torch.Tensor,
    center_variance: float,
    size_variance: float,
) -> torch.Tensor:
    """ Convert regressional location results of SSD into boxes in the form of
        (center_x, center_y, h, w)

    $$hat{center} * center_variance = \frac {center - center_prior} {hw_prior}$$
    $$exp(hat{hw} * size_variance) = \frac {hw} {hw_prior}$$

    :param locations: (batch_size, num_priors, 4) the regression output of SSD,
        containing the outputs
    :param priors: (num_priors, 4) or (batch_size/1, num_priors, 4) prior boxes
    :param center_variance: changes the scale of center
    :param size_variance: changes scale of size
    :return: priors [[center_x, center_y, w, h]], relative to the image size
    """
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)

    centers = locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2]
    hws = torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]

    return torch.cat([centers, hws], dim=locations.dim() - 1)


def convert_boxes_to_locations(
    boxes: torch.Tensor,
    priors: torch.Tensor,
    center_variance: float,
    size_variance: float,
) -> torch.Tensor:
    """ Convert boxes (x, y, w, h) to regressional location results of SSD

    $$hat{center} * center_variance = \frac {center - center_prior} {hw_prior}$$
    $$exp(hat{hw} * size_variance) = \frac {hw} {hw_prior}$$

    :param boxes: center form boxes
    :param priors: center form priors
    :param center_variance: changes the scale of center
    :param size_variance: changes scale of size
    :return: locations for training SSD
    """
    if priors.dim() + 1 == boxes.dim():
        priors = priors.unsqueeze(0)
    centers = (boxes[..., :2] - priors[..., :2]) / priors[..., 2:] / center_variance
    hws = torch.log(boxes[..., 2:] / priors[..., 2:]) / size_variance
    return torch.cat([centers, hws], dim=boxes.dim() - 1)


def center_bbox_to_corner_bbox(center_bboxes: torch.Tensor) -> torch.Tensor:
    """Convert x, y, w, h form to x1, y1, x2, y2."""
    point_1 = center_bboxes[..., :2] - center_bboxes[..., 2:] / 2
    point_2 = center_bboxes[..., :2] + center_bboxes[..., 2:] / 2
    return torch.cat([point_1, point_2], center_bboxes.dim() - 1)


def corner_bbox_to_center_bbox(corner_bboxes: torch.Tensor) -> torch.Tensor:
    """Convert x1, y1, x2, y2 form to x, y, w, h."""
    xy = (corner_bboxes[..., :2] + corner_bboxes[..., 2:]) / 2
    wh = corner_bboxes[..., 2:] - corner_bboxes[..., :2]
    return torch.cat([xy, wh], corner_bboxes.dim() - 1)
