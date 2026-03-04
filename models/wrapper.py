import torch
from torch import nn

class MultiCropWrapper(nn.Module):
    
    def __init__(self, backbone, ssl_head, reid_head=None, is_student=True):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.ssl_head = ssl_head
        self.reid_head = reid_head
        self.is_student = is_student

    def forward(self, x, vids=None):
        
        if self.training:
            # convert to list
            if not isinstance(x, list):
                x = [x]
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            num_crops = idx_crops[-1]
            start_idx, output = 0, torch.empty(0).to(x[0].device)
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                # accumulate outputs
                output = torch.cat((output, _out))
                start_idx = end_idx
            # Run the head forward on the concatenated features.
            
            if self.is_student:
                chunks = output.chunk(num_crops)
                reid_input = torch.cat([chunks[0], chunks[1]]) if num_crops > 1 else chunks[0]
                if self.ssl_head is not None:
                    return self.ssl_head(output), \
                        self.reid_head(reid_input), vids.repeat(2)
                else:
                    return output, \
                        self.reid_head(chunks[0])
            else:
                if self.ssl_head is not None:
                    return self.ssl_head(output)
                if self.reid_head is not None:
                    chunks = output.chunk(num_crops)
                    reid_input = torch.cat([chunks[0], chunks[1]]) \
                        if num_crops > 1 else chunks[0]
                    return self.reid_head(reid_input)
                else:
                    return output
        else:
            if isinstance(x, list):
                x = torch.cat(x[:2]) if len(x) > 1 else x[0]
            return self.reid_head(self.backbone(x))
