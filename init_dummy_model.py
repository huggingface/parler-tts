from stable_speech import StableSpeechConfig, StableSpeechForCausalLM, StableSpeechForConditionalGeneration, StableSpeechDecoderConfig
from transformers import T5Config, EncodecConfig


decoder_config = StableSpeechDecoderConfig(
    max_position_embeddings=1024,
    num_hidden_layers=2,
    ffn_dim=256,
    num_attention_heads=4,
    layerdrop=0.0,
    use_cache=True,
    activation_function="gelu",
    hidden_size=256,
    dropout=0.1,
    attention_dropout=0.1,
    activation_dropout=0.1,
)
# TODO: ?? how to make it stop ?
        

        
decoder = StableSpeechForCausalLM(decoder_config)

decoder.save_pretrained("/home/yoach/dataspeech/artefacts/decoder/")



model = StableSpeechForConditionalGeneration.from_sub_models_pretrained(
    text_encoder_pretrained_model_name_or_path="t5-base",
    audio_encoder_pretrained_model_name_or_path="facebook/encodec_32khz",
    decoder_pretrained_model_name_or_path="/home/yoach/dataspeech/artefacts/decoder/",
)

# set the appropriate bos/pad token ids
model.generation_config.decoder_start_token_id = 2048
model.generation_config.pad_token_id = 2048

# set other default generation config params
model.generation_config.max_length = int(30 * model.audio_encoder.config.frame_rate)
model.generation_config.do_sample = True
model.generation_config.guidance_scale = 3.0

model.save_pretrained("/home/yoach/dataspeech/artefacts/tiny-model/")