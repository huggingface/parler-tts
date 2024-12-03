from transformers import LogitsProcessor, LogitsProcessorList
from transformers.pytorch_utils import isin_mps_friendly
import math
import torch

class ParlerTTSLogitsProcessor(LogitsProcessor):
    r"""This processor ensures that the delayed pattern mask constraints are respected.

    <Tip warning={true}>

    This logits processor is exclusively compatible with Parler-TTS. 
    See the model documentation for examples.

    </Tip>

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        min_eos_p (`float`, *optional*):
            Minimum end of speech threshold.
    """

    def __init__(self, eos_token_id, num_codebooks: int, batch_size: int, device: str = "cpu"):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.num_codebooks = num_codebooks
        self.device = device


        self.codebook_idx = torch.arange(self.batch_size*self.num_codebooks, device=self.device)
        self.first_codebooks_unfinished = torch.arange(batch_size, device=device)*num_codebooks
        
        max_codebooks = torch.arange(self.batch_size, device=self.device)*self.num_codebooks + self.num_codebooks -1
        self.max_codebooks = max_codebooks
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        is_eos = isin_mps_friendly(input_ids, self.eos_token_id).sum(1)
        
        self.first_codebooks_unfinished = torch.where((is_eos[self.first_codebooks_unfinished]>0) & (self.first_codebooks_unfinished<self.max_codebooks), self.first_codebooks_unfinished+1, self.first_codebooks_unfinished)
                
        # every codebook higher than the first one unfinished will never be eos
        eos_token_mask = self.codebook_idx > self.first_codebooks_unfinished.repeat_interleave(self.num_codebooks)
        scores[eos_token_mask, self.eos_token_id] = -math.inf
        
        return scores