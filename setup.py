# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import setuptools


_deps = [
    "transformers>=4.46.1,<=4.46.1",
    "torch",
    "sentencepiece",
    "descript-audio-codec",
    "descript-audiotools @ git+https://github.com/descriptinc/audiotools", # temporary fix as long as 0.7.4 is not published
    "protobuf>=4.0.0"
]

_extras_dev_deps = [
    "black~=23.1",
    "isort>=5.5.4",
    "ruff>=0.0.241,<=0.0.259",
]

_extras_training_deps = [
    "jiwer",
    "wandb",
    "accelerate",
    "evaluate",
    "datasets[audio]>=2.14.5",
]

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# read version
with open(os.path.join(here, "parler_tts", "__init__.py"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break
    else:
        raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="parler_tts",
    version=version,
    description="Toolkit for using and training Parler-TTS, a high-quality text-to-speech model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=_deps,
    extras_require={
        "dev": _extras_dev_deps,
        "train": _extras_training_deps,
    },
)
