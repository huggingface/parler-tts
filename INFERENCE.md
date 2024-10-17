# Inference tips

Parler-TTS benefits from a number of optimizations that can make the model up to 4x faster. Add to this the ability to stream audio as it's being generated, and you can achieve time-to-first audio in under 500ms on a modern GPU.

## 📖 Quick Index
* [Efficient Attention Implementation](#efficient-attention-implementations)
* [Compilation](#compilation)
* [Streaming](#streaming)
* [Batch generation](#batch-generation)
* [Speaker Consistency](#speaker-consistency)

## Efficient Attention implementations

Parler-TTS supports [SDPA](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) and [Flash Attention 2](https://github.com/Dao-AILab/flash-attention).  

SDPA is used by default and speeds up generation time by up to 1.4x compared with eager attention.

To switch between attention implementations, simply specify `attn_implementation=attn_implementation` when loading the checkpoints:

```py
from parler_tts import ParlerTTSForConditionalGeneration

torch_device = "cuda:0" # use "mps" for Mac
torch_dtype = torch.bfloat16
model_name = "parler-tts/parler-tts-mini-v1"

attn_implementation = "eager" # "sdpa" or "flash_attention_2"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation=attn_implementation
).to(torch_device, dtype=torch_dtype)
```

## Compilation

[Compiling](https://pytorch.org/docs/stable/generated/torch.compile.html) the forward method of Parler can speed up generation time by up to 4.5x.

As an indication, `mode=default` brings a speed-up of 1.4 times compared to no compilation, while `mode="reduce-overhead"` brings much faster generation, at the cost of a longer compilation time and the need to generate twice to see the benefits of compilation.

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

torch_device = "cuda:0"
torch_dtype = torch.bfloat16
model_name = "parler-tts/parler-tts-mini-v1"

# need to set padding max length
max_length = 50

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
    attn_implementation="eager"
).to(torch_device, dtype=torch_dtype)

# compile the forward pass
compile_mode = "default" # chose "reduce-overhead" for 3 to 4x speed-up
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode=compile_mode)

# warmup
inputs = tokenizer("This is for compilation", return_tensors="pt", padding="max_length", max_length=max_length).to(torch_device)

model_kwargs = {**inputs, "prompt_input_ids": inputs.input_ids, "prompt_attention_mask": inputs.attention_mask, }

n_steps = 1 if compile_mode == "default" else 2
for _ in range(n_steps):
    _ = model.generate(**model_kwargs)


# now you can benefit from compilation speed-ups
...

```


## Streaming

### How Does It Work?

Parler-TTS is an auto-regressive transformer-based model, meaning generates audio codes (tokens) in a causal fashion.

At each decoding step, the model generates a new set of audio codes, conditional on the text input and all previous audio codes. From the 
frame rate of the [DAC model](https://huggingface.co/parler-tts/dac_44khZ_8kbps) used to decode the generated codes to audio waveform,  each set of generated audio codes corresponds to 0.011 seconds. This means we require a total of 1720 decoding steps to generate 20 seconds of audio.

Rather than waiting for the entire audio sequence to be generated, which would require the full 1720 decoding steps, we can start playing the audio after a specified number of decoding steps have been reached, a techinque known as [*streaming*](https://huggingface.co/docs/transformers/main/en/generation_strategies#streaming). 
For example, after 86 steps we have the first second of audio ready, and so can play this without waiting for the remaining decoding steps to be complete. As we continue to generate with the Parler-TTS model, we append new chunks of generated audio to our output waveform on-the-fly. After the full 1720 decoding steps, the generated audio is complete, and is composed of 20 chunks of audio, each corresponding to 86 tokens.
This method of playing incremental generations reduces the latency of the Parler-TTS model from the total time to generate 1720 tokens, to the time taken to play the first chunk of audio (86 tokens). This can result in significant improvements to perceived latency,  particularly when the chunk size is chosen to be small. In practice, the chunk size should be tuned to your device: using a smaller chunk size will mean that the first chunk is ready faster, but should not be chosen so small that the model generates slower than the time it takes to play the audio.


### How Can I Use It?

We've added [ParlerTTSStreamer](https://github.com/huggingface/parler-tts/blob/main/parler_tts/streamer.py) to the library. Don't hesitate to adapt it to your use-case.

Here's how to create a generator out of the streamer.

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread

torch_device = "cuda:0" # Use "mps" for Mac 
torch_dtype = torch.bfloat16
model_name = "parler-tts/parler-tts-mini-v1"

# need to set padding max length
max_length = 50

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
).to(torch_device, dtype=torch_dtype)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

def generate(text, description, play_steps_in_s=0.5):
  play_steps = int(frame_rate * play_steps_in_s)
  streamer = ParlerTTSStreamer(model, device=torch_device, play_steps=play_steps)
  # tokenization
  inputs = tokenizer(description, return_tensors="pt").to(torch_device)
  prompt = tokenizer(text, return_tensors="pt").to(torch_device)
  # create generation kwargs
  generation_kwargs = dict(
    input_ids=inputs.input_ids,
    prompt_input_ids=prompt.input_ids,
    attention_mask=inputs.attention_mask,
    prompt_attention_mask=prompt.attention_mask,
    streamer=streamer,
    do_sample=True,
    temperature=1.0,
    min_new_tokens=10,
  )
  # initialize Thread
  thread = Thread(target=model.generate, kwargs=generation_kwargs)
  thread.start()
  # iterate over chunks of audio
  for new_audio in streamer:
    if new_audio.shape[0] == 0:
      break
    print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 4)} seconds")
    yield sampling_rate, new_audio


# now you can do
text = "This is a test of the streamer class"
description = "Jon's talking really fast."

chunk_size_in_s = 0.5

for (sampling_rate, audio_chunk) in generate(text, description, chunk_size_in_s):
  # You can do everything that you need with the chunk now
  # For example: stream it, save it, play it.
  print(audio_chunk.shape) 
```

## Batch generation

Batching means combining operations for multiple samples to bring the overall time spent generating the samples lower than generating sample per sample.

Here is a quick example of how you can use it:

```py
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import scipy


repo_id = "parler-tts/parler-tts-mini-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

input_text = ["Hey, how are you doing?", "I'm not sure how to feel about it."]
description = 2 * ["A male speaker with a monotone and high-pitched voice is delivering his speech at a really low speed in a confined environment."]

inputs = tokenizer(description, return_tensors="pt", padding=True).to("cuda")
prompt = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")

set_seed(0)
generation = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    prompt_input_ids=prompt.input_ids,
    prompt_attention_mask=prompt.attention_mask,
    do_sample=True,
    return_dict_in_generate=True,
)

audio_1 = generation.sequences[0, :generation.audios_length[0]]
audio_2 = generation.sequences[1, :generation.audios_length[1]]

print(audio_1.shape, audio_2.shape)
scipy.io.wavfile.write("sample_out.wav", rate=feature_extractor.sampling_rate, data=audio_1.cpu().numpy().squeeze())
scipy.io.wavfile.write("sample_out_2.wav", rate=feature_extractor.sampling_rate, data=audio_2.cpu().numpy().squeeze())
```

## Speaker Consistency

The checkpoint was trained on 34 speakers. The full list of available speakers includes:
Laura, Gary, Jon, Lea, Karen, Rick, Brenda, David, Eileen, Jordan, Mike, Yann, Joy, James, Eric, Lauren, Rose, Will, Jason, Aaron, Naomie, Alisa, Patrick, Jerry, Tina, Jenna, Bill, Tom, Carol, Barbara, Rebecca, Anna, Bruce, and Emily.

However, the models performed better with certain speakers. Below are the top 20 speakers for each model variant, ranked by their average speaker similarity scores:

### Large Model - Top 20 Speakers

| Speaker | Similarity Score |
|---------|------------------|
| Will    | 0.906055         |
| Eric    | 0.887598         |
| Laura   | 0.877930         |
| Alisa   | 0.877393         |
| Patrick | 0.873682         |
| Rose    | 0.873047         |
| Jerry   | 0.871582         |
| Jordan  | 0.870703         |
| Lauren  | 0.867432         |
| Jenna   | 0.866455         |
| Karen   | 0.866309         |
| Rick    | 0.863135         |
| Bill    | 0.862207         |
| James   | 0.856934         |
| Yann    | 0.856787         |
| Emily   | 0.856543         |
| Anna    | 0.848877         |
| Jon     | 0.848828         |
| Brenda  | 0.848291         |
| Barbara | 0.847998         |

### Mini Model - Top 20 Speakers

| Speaker | Similarity Score |
|---------|------------------|
| Jon     | 0.908301         |
| Lea     | 0.904785         |
| Gary    | 0.903516         |
| Jenna   | 0.901807         |
| Mike    | 0.885742         |
| Laura   | 0.882666         |
| Lauren  | 0.878320         |
| Eileen  | 0.875635         |
| Alisa   | 0.874219         |
| Karen   | 0.872363         |
| Barbara | 0.871509         |
| Carol   | 0.863623         |
| Emily   | 0.854932         |
| Rose    | 0.852246         |
| Will    | 0.851074         |
| Patrick | 0.850977         |
| Eric    | 0.845459         |
| Rick    | 0.845020         |
| Anna    | 0.844922         |
| Tina    | 0.839160         |

The numbers represent the average speaker similarity between a random snippet of the person speaking and a randomly Parler-generated snippet. Higher scores indicate better model performance in maintaining voice consistency.

These scores are derived from [dataset for Mini](https://huggingface.co/datasets/ylacombe/parler-tts-mini-v1_speaker_similarity) and [dataset for Large](https://huggingface.co/datasets/ylacombe/parler-large-v1-og_speaker_similarity).