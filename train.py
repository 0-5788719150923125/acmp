# Copied from here: https://huggingface.co/docs/transformers/en/tasks/text-to-speech
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from datasets import Audio, load_dataset, load_from_disk
from speechbrain.inference.classifiers import EncoderClassifier
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = "data"


device = "cpu"
cache_dir = "data"


resume = True


checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(
    checkpoint,
    cache_dir=cache_dir,
    trust_remote_code=True,
    device=device,
    device_map=device,
)

tokenizer = processor.tokenizer


if not resume:

    dataset = load_dataset(
        "facebook/voxpopuli",
        "nl",
        split="train",
        trust_remote_code=True,
    )

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def extract_all_chars(batch):
        all_text = " ".join(batch["normalized_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )

    dataset_vocab = set(vocabs["vocab"][0])
    tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}

    replacements = [
        ("à", "a"),
        ("ç", "c"),
        ("è", "e"),
        ("ë", "e"),
        ("í", "i"),
        ("ï", "i"),
        ("ö", "o"),
        ("ü", "u"),
    ]

    def cleanup_text(inputs):
        for src, dst in replacements:
            inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
        return inputs

    dataset = dataset.map(cleanup_text)

    speaker_counts = defaultdict(int)

    for speaker_id in dataset["speaker_id"]:
        speaker_counts[speaker_id] += 1

    def select_speaker(speaker_id):
        return 100 <= speaker_counts[speaker_id] <= 400

    dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

    spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

    speaker_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device, "device_map": device},
        savedir=os.path.join("/tmp", spk_model_name),
    )

    def create_speaker_embedding(waveform):
        with torch.no_grad():
            speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
            speaker_embeddings = torch.nn.functional.normalize(
                speaker_embeddings, dim=2
            )
            speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
        return speaker_embeddings

    def prepare_dataset(example):
        audio = example["audio"]

        example = processor(
            text=example["normalized_text"],
            audio_target=audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=False,
        )

        # strip off the batch dimension
        example["labels"] = example["labels"][0]

        # use SpeechBrain to obtain x-vector
        example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

        return example

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

    def is_not_too_long(input_ids):
        input_length = len(input_ids)
        return input_length < 200

    dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])

    dataset = dataset.train_test_split(test_size=0.1)
    dataset.save_to_disk(f"{cache_dir}/processed_dataset")
else:
    dataset = load_from_disk(f"{cache_dir}/processed_dataset")


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch


data_collator = TTSDataCollatorWithPadding(processor=processor)


model = SpeechT5ForTextToSpeech.from_pretrained(
    checkpoint, cache_dir=cache_dir, device_map=device
)
model.config.use_cache = False


training_args = Seq2SeqTrainingArguments(
    output_dir=f"{cache_dir}/speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=False,
    # deepspeed="deepspeed.json",
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)


trainer.train()
processor.save_pretrained("acmp")
