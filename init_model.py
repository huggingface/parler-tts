from stable_speech import StableSpeechConfig, StableSpeechForCausalLM, StableSpeechForConditionalGeneration, StableSpeechDecoderConfig
from transformers import T5Config, EncodecConfig
from transformers import AutoConfig

decoder_config = StableSpeechDecoderConfig(
    max_position_embeddings=2048,
    num_hidden_layers=24,
    ffn_dim=4096,
    num_attention_heads=16,
    layerdrop=0.0,
    use_cache=True,
    activation_function="gelu",
    hidden_size=1024,
    dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
)        

        
decoder = StableSpeechForCausalLM(decoder_config)
decoder.save_pretrained("/home/yoach/dataspeech/artefacts/decoder/")


t5 = AutoConfig.from_pretrained("t5-base")

model = StableSpeechForConditionalGeneration.from_sub_models_pretrained(
    text_encoder_pretrained_model_name_or_path="t5-base",
    audio_encoder_pretrained_model_name_or_path="facebook/encodec_32khz",
    decoder_pretrained_model_name_or_path="/home/yoach/dataspeech/artefacts/decoder/",
    vocab_size = t5.vocab_size
)

# set the appropriate bos/pad token ids
model.generation_config.decoder_start_token_id = 2049
model.generation_config.pad_token_id = 2048
model.generation_config.eos_token_id = 2048

# set other default generation config params
model.generation_config.max_length = int(30 * model.audio_encoder.config.frame_rate)
model.generation_config.do_sample = False # True
model.generation_config.guidance_scale = 1 # 3.0

model.save_pretrained("/home/yoach/dataspeech/artefacts/small-stable-speech-untrained/")