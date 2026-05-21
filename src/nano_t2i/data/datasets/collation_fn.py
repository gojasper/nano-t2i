from typing import Dict, List, Optional, Union

import numpy as np
import torch


def custom_collation_fn(
    samples: List[Dict[str, Union[int, float, np.ndarray, torch.Tensor]]],
    combine_tensors: bool = True,
    combine_scalars: bool = True,
    allow_padding: bool = True,
    max_length: Optional[int] = None,
) -> dict:
    """
    Collate function for PyTorch DataLoader.

    Args:
        samples(List[Dict[str, Union[int, float, np.ndarray, torch.Tensor]]]): List of samples.
        combine_tensors (bool): Whether to turn lists of tensors into a single tensor.
        combine_scalars (bool): Whether to turn lists of scalars into a single ndarray.
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}
    for s in samples:
        [batched[key].append(s[key]) for key in batched]
    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                try:
                    result[key] = torch.stack(list(batched[key]))
                except Exception as e:
                    if allow_padding:
                        # pad to the longest tensor in dim 0
                        if max_length is None:
                            max_length = max(tensor.shape[0] for tensor in batched[key])
                        else:
                            max_length = min(
                                max_length,
                                max(tensor.shape[0] for tensor in batched[key]),
                            )
                        print(f"Max length: {max_length}")
                        padded_tensors = []
                        for tensor in batched[key]:
                            if tensor.shape[0] < max_length:
                                tensor = torch.nn.functional.pad(
                                    tensor, (0, 0, 0, max_length - tensor.shape[0])
                                )
                            else:
                                tensor = tensor[:max_length]
                            padded_tensors.append(tensor)
                        result[key] = torch.stack(padded_tensors)
                        print(f"Padded tensor {key} shape: {result[key].shape}")
                    else:
                        raise e
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])

    del samples
    del batched
    return result
