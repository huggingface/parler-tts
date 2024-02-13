import csv
import os
import sys
from dataclasses import dataclass, field
from transformers import HfArgumentParser


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
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    private_repo: bool = field(
        default=True,
        metadata={"help": "Whether or not to push the processed dataset to a private repository on the Hub"},
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

    # 3. Iterate over dev/test files
    for split in ["dev", "test"]:
        data_dir = os.path.join(data_args.dataset_dir, split)
        metadata = os.path.join(data_dir, "stm")

        with open(metadata, "r") as file:
            for line in file:
                # example line is: 'EDACC-C06 1 EDACC-C06-A 0.00 5.27 <male,l1> C ELEVEN DASH P ONE\n
                # the transcription always comes to the right of the last rangle bracket
                text_idx = line.rfind(">") + 1
                text = line[text_idx:-1]
                # the metadata immediately proceeds this
                line = line[:text_idx]
                file, channel, speaker, start, end, gender = line.split(" ")





if __name__ == "__main__":
    main()
