import torch
from dac.model import DAC
from transformers import PreTrainedModel
from transformers.models.encodec.modeling_encodec import EncodecDecoderOutput, EncodecEncoderOutput

from .configuration_dac import DACConfig


# model doesn't support batching yet


class DACModel(PreTrainedModel):
    config_class = DACConfig

    # Set main input to 'input_values' for voice steering
    main_input_name = "input_values"

    def __init__(self, config):
        super().__init__(config)

        self.model = DAC(
            n_codebooks=config.num_codebooks,
            latent_dim=config.latent_dim,
            codebook_size=config.codebook_size,
        )

    def encode(
        self, input_values, padding_mask=None, bandwidth=None, return_dict=None, n_quantizers=None, sample_rate=None
    ):
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                Not used, kept to have the same inferface as HF encodec.
            n_quantizers (`int`, *optional*) :
                Number of quantizers to use, by default None
                If None, all quantizers are used.
            sample_rate (`int`, *optional*) :
                Signal sampling_rate

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
            `codebook` of shape `[batch_size, num_codebooks, frames]`.
            Scale is not used here.

        """
        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        audio_data = self.model.preprocess(input_values, sample_rate)

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # TODO: for now, no chunk length

        chunk_length = None  # self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.config.chunk_stride

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames = []
        scales = []

        step = chunk_length - stride
        if (input_length % stride) - step != 0:
            raise ValueError(
                "The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly."
            )

        for offset in range(0, input_length - step, stride):
            mask = padding_mask[..., offset : offset + chunk_length].bool()
            frame = audio_data[:, :, offset : offset + chunk_length]

            scale = None

            _, encoded_frame, _, _, _ = self.model.encode(frame, n_quantizers=n_quantizers)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = torch.stack(encoded_frames)

        if not return_dict:
            return (encoded_frames, scales)

        return EncodecEncoderOutput(encoded_frames, scales)

    def decode(
        self,
        audio_codes,
        audio_scales,
        padding_mask=None,
        return_dict=None,
    ):
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Not used, kept to have the same inferface as HF encodec.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
                Not used yet, kept to have the same inferface as HF encodec.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict or self.config.return_dict

        # TODO: for now, no chunk length

        if len(audio_codes) != 1:
            raise ValueError(f"Expected one frame, got {len(audio_codes)}")

        audio_values = self.model.quantizer.from_codes(audio_codes.squeeze(0))[0]
        audio_values = self.model.decode(audio_values)
        if not return_dict:
            return (audio_values,)
        return EncodecDecoderOutput(audio_values)

    def forward(self, tensor):
        raise ValueError("`DACModel.forward` not implemented yet")
