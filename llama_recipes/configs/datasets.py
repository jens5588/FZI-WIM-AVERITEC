# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/llama_datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/llama_datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/llama_datasets/alpaca_data.json"


@dataclass
class averitec_question_dataset:
    dataset: str = "averitec_question_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class averitec_verification_dataset:
    dataset: str = "averitec_verification_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class averitec_qa_dataset:
    dataset: str = "averitec_qa_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
