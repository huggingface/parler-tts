
from transformers import PretrainedConfig


class DACConfig(PretrainedConfig):
    model_type = "dac"

    def __init__(
        self,
        num_codebooks: int = 9,
        model_bitrate: int = 8,  # kbps
        codebook_size: int = 1024,
        latent_dim: int = 1024,
        frame_rate: int = 86,
        sampling_rate: int = 44100,
        **kwargs,
    ):
        self.codebook_size = codebook_size
        self.model_bitrate = model_bitrate
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.frame_rate = frame_rate
        self.sampling_rate = sampling_rate

        super().__init__(**kwargs)
