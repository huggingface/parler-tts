from parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration, ParlerTTSDecoderConfig
from transformers import AutoConfig
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_directory", type=str, help="Directory where to save the model and the decoder.")
    parser.add_argument("--text_model", type=str, help="Repository id or path to the text encoder.")
    parser.add_argument("--audio_model", type=str, help="Repository id or path to the audio encoder.")

    args = parser.parse_args()

    text_model = args.text_model
    encodec_version = args.audio_model

    t5 = AutoConfig.from_pretrained(text_model)
    encodec = AutoConfig.from_pretrained(encodec_version)

    encodec_vocab_size = encodec.codebook_size
    num_codebooks = encodec.num_codebooks
    print("num_codebooks", num_codebooks)

    decoder_config = ParlerTTSDecoderConfig(
        vocab_size=encodec_vocab_size + 64,  # + 64 instead of +1 to have a multiple of 64
        max_position_embeddings=4096,  # 30 s = 2580
        num_hidden_layers=30,
        ffn_dim=6144,
        num_attention_heads=24,
        num_key_value_heads=24,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1536,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        pad_token_id=encodec_vocab_size,
        eos_token_id=encodec_vocab_size,
        bos_token_id=encodec_vocab_size + 1,
        num_codebooks=num_codebooks,
    )

    decoder = ParlerTTSForCausalLM(decoder_config)
    decoder.save_pretrained(os.path.join(args.save_directory, "decoder"))

    model = ParlerTTSForConditionalGeneration.from_sub_models_pretrained(
        text_encoder_pretrained_model_name_or_path=text_model,
        audio_encoder_pretrained_model_name_or_path=encodec_version,
        decoder_pretrained_model_name_or_path=os.path.join(args.save_directory, "decoder"),
        vocab_size=t5.vocab_size,
    )

    # set the appropriate bos/pad token ids
    model.generation_config.decoder_start_token_id = encodec_vocab_size + 1
    model.generation_config.pad_token_id = encodec_vocab_size
    model.generation_config.eos_token_id = encodec_vocab_size

    # set other default generation config params
    model.generation_config.max_length = int(30 * model.audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True  # True


    model.config.pad_token_id = encodec_vocab_size
    model.config.decoder_start_token_id = encodec_vocab_size + 1

    model.save_pretrained(os.path.join(args.save_directory, "parler-tts-untrained-larger/"))
