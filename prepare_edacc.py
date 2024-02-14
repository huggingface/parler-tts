import csv
import os
import re
import shutil
import sys
from dataclasses import dataclass, field

from datasets import DatasetDict, Dataset, Audio
from tqdm import tqdm
from transformers import HfArgumentParser
import soundfile as sf


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our data for prepareation
    """
    dataset_dir: str = field(
        default=None,
        metadata={
            "help": "Path where the EdAcc tar.gz archive is extracted. Leave in it's raw format: the script will "
                    "assume it's unchanged from the download and use relative paths to load the relevant audio files."
        }
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": "Where to save the processed dataset to disk. If unspecified, uses a 'pretty' version of the "
            "original dataset name. E.g. 'facebook/voxpopuli' will be saved under 'voxpopuli'."
        },
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    hub_dataset_id: str = field(
        default=False,
        metadata={"help": "Repository namespace if pushing to the Hugging Face Hub."},
    )
    private_repo: bool = field(
        default=True,
        metadata={"help": "Whether or not to push the processed dataset to a private repository on the Hub"},
    )
    max_samples: int = field(
        default=None,
        metadata={"help": "Maximum number of samples per split. Useful for debugging purposes."},
    )

ACCENT_MAPPING = {
    'Italian': 'Italian',
    'International': 'Unknown',
    'American': 'American',
    'English': 'English',
    'Latin American': 'Latin American',
    'British': 'English',
    'Romanian': 'Romanian',
    'Standard Indian English': 'Indian',
    'Trans-Atlantic': 'Unknown',
    'Slightly American': 'American',
    'European': 'Unknown',
    'Scottish (Fife)': 'Scottish',
    'English with Scottish inflections': 'Scottish',
    'Indian': 'Indian',
    'Asian': 'Asian',
    'NA': 'Unknown',
    'German': 'German',
    'South London': 'English',
    'Dutch': 'Dutch',
    'Mostly West Coast American with some Australian Intonation': 'American',
    'Japanese': 'Japanese',
    'Chinese': 'Chinese',
    'Generic middle class white person': 'English',
    'French': 'French',
    'Chinese accent or mixed accent(US, UK, China..) perhaps': 'Chinese',
    'American accent': 'American',
    'Catalan': 'Catalan',
    'American, I guess.': 'American',
    'Spanish American': 'Latin American',
    'Spanish': 'Spanish',
    'Standard American,Scottish': 'American',
    'Bulgarian': 'Bulgarian',
    'Latin': 'Latin American',
    'Latín American': 'Latin American',
    'Mexican': 'Latin American', # TODO: un-generalise latin american accents?
    'North American': 'American',
    'Afrian': 'African',
    'Nigerian': 'African', # TODO: un-generalise african accents?
    'East-European': 'Eastern European',
    'Eastern European': 'Eastern European',
    'Southern London': 'English',
    'American with a slight accent': 'American',
    'American-ish': 'American',
    'Indian / Pakistani accent': 'Indian',
    'Pakistani/American': 'Pakistani',
    'African accent': 'African',
    'Kenyan': 'African',  # TODO: un-generalise african accents?
    'Ghanaian': 'African', # TODO: un-generalise african accents?
    'Spanish accent': 'Spanish',
    'Lithuanian': 'Lithuanian',
    'Lithuanian (eastern European)': 'Lithuanian',
    'Indonesian': 'Indonesian',
    'Egyptian': 'Egyptian',
    'South African English': 'South African',
    "Neutral": "English",
    'Neutral accent': 'English',
    'Neutral English, Italian': 'English',
    'Fluent': 'Unknown',
    'Glaswegian': 'Scottish',
    'Glaswegian (not slang)': 'Scottish',
    'Irish': 'Irish',
    'Jamaican': 'Jamaican',
    'Jamaican accent': 'Jamaican',
    'Irish/ Dublin': 'Irish',
    'South Dublin Irish': 'Irish',
    'italian': 'Italian',
    'italian mixed with American and British English': 'Italian',
    'Italian mixed with American accent': 'Italian',
    'South American': 'Latin American',
    'Brazilian accent': 'Latin American', # TODO: un-generalise latin american accents?
    'Israeli': 'Israeli',
    'Vietnamese accent': 'Vietnamese',
    'Southern Irish': 'Irish',
    'Slight Vietnamese accent': 'Vietnamese',
    'Midwestern United States': 'American',
    'Vietnamese English': 'Vietnamese',
    "Vietnamese": "Vietnamese",
    "": "Unknown"
}


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser(DataTrainingArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]

    # 1. Load accents for each speaker
    linguistic_background = dict()
    linguistic_background_csv = os.path.join(data_args.dataset_dir, "linguistic_background.csv")
    with open(linguistic_background_csv, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",")
        for line in reader:
            linguistic_background[line["PARTICIPANT_ID"]] = line["How would you describe your accent in English? (e.g. Italian, Glaswegian)"]

    # 2. Clean accents for each speaker
    linguistic_background_clean = {participant: ACCENT_MAPPING[accent.strip()] for participant, accent in linguistic_background.items()}

    # 3. Initialize dataset dict
    raw_datasets = DatasetDict()

    if data_args.overwrite_output_dir and os.path.exists(data_args.output_dir) and os.path.isdir(data_args.output_dir):
        shutil.rmtree(data_args.output_dir)
    output_dir_processed = os.path.join(data_args.output_dir, "processed")

    # 4. Iterate over dev/test files
    for split, split_formatted in zip(["dev", "test"], ["validation", "test"]):
        data_dir = os.path.join(data_args.dataset_dir, split)
        metadata = os.path.join(data_dir, "stm")
        output_dir_split = os.path.join(output_dir_processed, split)
        os.makedirs(output_dir_split, exist_ok=True)

        all_speakers = []
        all_genders = []
        all_l1s = []
        all_texts = []
        all_audio_paths = []
        all_normalized_accents = []
        all_raw_accents = []
        
        current_audio = None
        current_audio_array = None
        current_sampling_rate = None
        current_counter = 1

        gender_pat = r'.*?\<(.*),.*'
        l1_pat = r'.*?\,(.*)>.*'

        with open(metadata, "r") as file:
            for idx, line in tqdm(enumerate(file), desc=split):
                # example line is: 'EDACC-C06 1 EDACC-C06-A 0.00 5.27 <male,l1> C ELEVEN DASH P ONE\n
                # the transcription always comes to the right of the last rangle bracket
                text_idx = line.find(">") + 1
                all_texts.append(line[text_idx + 1:-1])
                # the metadata immediately proceeds this
                line = line[:text_idx]
                file, channel, speaker, start, end, gender_l1 = line.split(" ")

                # add speaker information to cumulative lists
                all_raw_accents.append(linguistic_background[speaker])
                all_normalized_accents.append(linguistic_background_clean[speaker])
                all_speakers.append(speaker)

                # add gender/l1 information
                all_genders.append(re.search(gender_pat, gender_l1).group(1))
                all_l1s.append(re.search(l1_pat, gender_l1).group(1))
                
                # read audio file if different from previous
                if file != current_audio:
                    current_audio_array, current_sampling_rate = sf.read(os.path.join(data_args.dataset_dir, "data", file + ".wav"))
                    current_audio = file
                    current_counter = 1
                else:
                    current_counter += 1

                # chunk audio file according to start/end times
                start = int(float(start) * current_sampling_rate)
                end = int(float(end) * current_sampling_rate)
                end = min(end, len(current_audio_array))
                chunked_audio = current_audio_array[start: end]
                save_path = os.path.join(output_dir_split, f"{file}-{current_counter}.wav")
                sf.write(save_path, chunked_audio, current_sampling_rate)
                all_audio_paths.append(save_path)

                if data_args.max_samples is not None and (data_args.max_samples - 1) == idx:
                    break

        raw_datasets[split_formatted] = Dataset.from_dict(
            {"speaker": all_speakers,
             "text": all_texts,
             "accent": all_normalized_accents,
             "raw_accent": all_raw_accents,
             "gender": all_genders,
             "language": all_l1s,
             "audio": all_audio_paths,
             }
        ).cast_column("audio", Audio())

    if data_args.push_to_hub:
        raw_datasets.push_to_hub(data_args.hub_dataset_id, token=True)

    raw_datasets.save_to_disk(data_args.output_dir)

if __name__ == "__main__":
    main()
