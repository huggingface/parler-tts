# Training Parler-TTS

This sub-folder contains all the information to train or finetune you own Parler-TTS model.






# Init model
python helpers/model_init_scripts/init_dummy_model.py /raid/yoach/artefacts/dummy_model/ "google-t5/t5-small" "ylacombe/dac_44khZ_8kbps"

text_model = "google-t5/t5-small"
encodec_version = "ylacombe/dac_44khZ_8kbps"
text_model = "google-t5/t5-small"
encodec_version = "facebook/encodec_24khz"
text_model = "google/flan-t5-base"
encodec_version = "ylacombe/dac_44khZ_8kbps"



## TODOs
- [ ] Add PEFT compatibility to do Lora fine-tuning.
- [ ] Enrich dataset with accent classifier