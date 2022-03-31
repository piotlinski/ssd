"""Additional training metrics."""
from typing import List, Tuple

import torch
from pytorch_lightning.metrics import Metric

from pytorch_ssd.data.bboxes import iou


class MeanAveragePrecision(Metric):
    """Mean average precision for object detection."""

    def __init__(
        self,
        iou_threshold: float,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.iou_threshold = iou_threshold
        self.epsilon = 1e-6

        self.add_state("ap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("ap_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        predictions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ground_truth: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ):
        """Set new value of mean average precision with predictions and target."""
        average_precisions = []

        for (boxes, scores, labels), (gt_boxes, _, gt_labels) in zip(
            predictions, ground_truth
        ):
            if gt_boxes.shape[0] == 0:
                continue

            sorted_scores, sort_indices = scores.sort(dim=-1, descending=True)
            sorted_boxes = boxes[sort_indices.unsqueeze(-1).expand_as(boxes)].view(
                -1, boxes.shape[-1]
            )
            sorted_labels = labels[sort_indices.expand_as(labels)]

            tps = torch.zeros(boxes.shape[0])
            fps = torch.zeros(boxes.shape[0])
            gt_box_used = torch.zeros(gt_boxes.shape[0])

            for idx, (box, label) in enumerate(zip(sorted_boxes, sorted_labels)):
                idx_mask = (gt_labels == label).nonzero(as_tuple=False).squeeze(1)

                masked_gt_boxes = gt_boxes[idx_mask]
                if masked_gt_boxes.numel() == 0:
                    continue

                ious = iou(masked_gt_boxes, box)
                max_iou, max_iou_idx = torch.max(ious, dim=-1)

                if max_iou > self.iou_threshold:
                    if gt_box_used[max_iou_idx] == 0:
                        tps[idx] = 1
                        gt_box_used[max_iou_idx] = 1
                    else:
                        fps[idx] = 1
                else:
                    fps[idx] = 1

            tps_cumsum = torch.cumsum(tps, dim=0)
            fps_cumsum = torch.cumsum(fps, dim=0)
            recalls = tps_cumsum / (gt_boxes.shape[0] + self.epsilon)
            precisions = tps_cumsum / (tps_cumsum + fps_cumsum + self.epsilon)
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))
            average_precisions.append(torch.trapz(precisions, recalls))

        self.ap_sum += sum(average_precisions)
        self.ap_total += len(average_precisions)

    def compute(self):
        """Compute current mean average precision value."""
        return self.ap_sum.float() / self.ap_total
