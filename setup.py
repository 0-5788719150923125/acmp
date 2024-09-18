from setuptools import find_packages, setup

setup(
    name="acmp",
    packages=find_packages(),
    install_requires=[
        "deepspeed",
        "numpy",
        "pydub",
        "scipy",
        "torch",
        "torchaudio",
        "transformers",
        "wave",
        "datasets",
        "soundfile",
        "speechbrain",
        "accelerate",
        "librosa",
    ],
)
