import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import sys

# device = "mps" if torch.backends.mps.is_available() else torch.device("cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

# Function to load a script from the 'scripts' folder
def load_script(file_name):
    script_path = os.path.join("scripts", file_name)
    with open(script_path, "r") as file:
        script = file.read().strip()
    return script

# Get script name from command line argument
if len(sys.argv) < 2:
    print("Please provide a script file name.")
    sys.exit(1)

script_file = sys.argv[1]  # First argument from the command line, e.g., 'your_script.txt'
script_name = os.path.splitext(script_file)[0]  # Remove .txt extension

# Create a folder with the script name (if it doesn't exist)
output_folder = os.path.join("output", script_name)
os.makedirs(output_folder, exist_ok=True)

# Load the script
prompt = load_script(script_file)

# Define the description for the desired voice
description = ("A soothing, mesmerizing, and mystical Indian female voice with soft tones, "
               "gentle delivery, and a captivating presence. The voice should evoke calmness "
               "and wonder, perfect for spiritual or ethereal stories. High quality and clear.")

# Convert the description and prompt to input IDs
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Create an attention mask
attention_mask = torch.ones(input_ids.shape, device=device)

# Generate the speech
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids, attention_mask=attention_mask)
audio_arr = generation.cpu().numpy().squeeze()

# Save the output as a .wav file in the corresponding folder
output_file = os.path.join(output_folder, f"{script_name}.wav")
sf.write(output_file, audio_arr, model.config.sampling_rate)

print(f"Audio generated and saved as {output_file}")