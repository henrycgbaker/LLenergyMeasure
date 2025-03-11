import torch.nn as nn

# -----------------------------------------------------------------------------
# For the FLOP counter:
# -----------------------------------------------------------------------------

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids):
        return self.model(input_ids=input_ids)
