from typing import List, Optional

import torch
def classification_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, top_k: Optional[List[int]] = None
) -> List[float]:
    """
    Computes the top-k classification accuracy provided with
    un-normalized logits of a model and ground truth targets.
    If top_k is not provided, defaults to top_1 accuracy.
    If top_k is provided as a list, then the values are sorted
    in ascending order.
    Args:
        logits: Un-normalized logits of a model. Softmax will be
            applied to these logits prior to computation of accuracy.
        targets: Vector of integers which represent indices of class
            labels.
        top_k: Optional list of integers in the range [1, max_classes].
    Returns:
        A list of length `top_k`, where each value represents top_i
        accuracy (i in `top_k`).
    """
    if top_k is None:
        top_k = [1]
    max_k = max(top_k)

    with torch.no_grad():
        _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
        predictions = predictions.t()
        correct = predictions.eq(targets.view(1, -1)).expand_as(predictions)

        results = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().mean().to('cpu').numpy()
            results.append(correct_k)

    return results
