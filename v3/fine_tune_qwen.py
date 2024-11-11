import collections
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator, DataCollatorWithPadding
)
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm
import evaluate
import os
from typing import Dict, List, Tuple


class HebrewQADataset:
    def __init__(self, json_file: str):
        """Initialize dataset from JSON file."""
        self.examples = []
        self._load_and_process_json(json_file)

    def _load_and_process_json(self, json_file: str):
        """Load and process JSON data into QA format."""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    # Skip impossible questions if marked as such
                    if qa.get('is_impossible', 'FALSE').upper() == 'TRUE':
                        continue

                    # Get the first answer (assuming it's the best one)
                    answer = qa['answers'][0]

                    self.examples.append({
                        'id': qa['id'],
                        'context': context,
                        'question': qa['question'],
                        'answers': {
                            'text': [answer['text']],
                            'answer_start': [answer['answer.start']]
                        }
                    })

    def to_huggingface_dataset(self) -> HFDataset:
        """Convert to HuggingFace Dataset format."""
        return HFDataset.from_list(self.examples)


def preprocess_function(examples: Dict, tokenizer, max_length: int = 384, stride: int = 128):
    """Preprocess examples for the model."""
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # Tokenize questions and contexts
    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        stride=stride,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Keep track of example IDs
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    # Add example IDs and initialize positions
    tokenized["example_id"] = []
    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_idx])

        # Get sequence IDs to identify which tokens are part of the context
        sequence_ids = tokenized.sequence_ids(i)

        # Get the context start index (first token after the question)
        context_start = 0
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id == 1:  # 1 indicates context tokens
                context_start = idx
                break

        # Get answer for this example
        answer = examples["answers"][sample_idx]
        start_char = answer["answer_start"][0]
        answer_text = answer["text"][0]
        end_char = start_char + len(answer_text)

        # Find token indices relative to the context start
        start_position = context_start  # Default to start of context
        end_position = context_start  # Default to start of context

        # Map character positions to token positions
        for idx, (token_start, token_end) in enumerate(offset_mapping[i]):
            if sequence_ids[idx] != 1:
                continue
            if token_start <= start_char and token_end > start_char:
                start_position = idx
            if token_start < end_char and token_end >= end_char:
                end_position = idx
                break

        tokenized["start_positions"].append(start_position)
        tokenized["end_positions"].append(end_position)

    return tokenized


def postprocess_qa_predictions(start_logits, end_logits, examples, features, tokenizer, n_best_size=20):
    """Post-process QA predictions to get actual text answers."""
    all_predictions = []

    for i in range(len(start_logits)):
        start_logit = start_logits[i]
        end_logit = end_logits[i]

        # Get the most likely start and end positions
        start_idx = int(np.argmax(start_logit))
        end_idx = int(np.argmax(end_logit))

        # Ensure valid span (end comes after start)
        if end_idx < start_idx:
            end_idx = start_idx

        # Get the actual text from the tokens
        tokens = features[i]
        token_ids = tokens["input_ids"]

        # Get the predicted span
        predicted_text = tokenizer.decode(token_ids[start_idx:end_idx + 1], skip_special_tokens=True)

        all_predictions.append({
            "prediction_text": predicted_text,
            "id": examples[i]["id"]
        })

    return all_predictions


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    metric = evaluate.load("squad")

    predictions, labels = eval_pred
    start_logits, end_logits = predictions

    # Get references from the evaluation dataset
    references = [
        {
            "id": ex["id"],
            "answers": {
                "text": ex["answers"]["text"],
                "answer_start": ex["answers"]["answer_start"]
            }
        }
        for ex in eval_pred.inputs
    ]

    # Process predictions
    predicted_answers = []
    for i, (start_logit, end_logit) in enumerate(zip(start_logits, end_logits)):
        # Get the most likely positions
        start_idx = np.argmax(start_logit)
        end_idx = np.argmax(end_logit)

        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx

        predicted_answers.append({
            "id": references[i]["id"],
            "prediction_text": "dummy"  # We'll replace this with actual text in production
        })

    # Compute metrics
    results = metric.compute(predictions=predicted_answers, references=references)

    return {
        "exact_match": round(results["exact_match"], 4),
        "f1": round(results["f1"], 4)
    }


def train(
        model_name: str,
        train_dataset: HFDataset,
        eval_dataset: HFDataset,
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 2e-5
):
    """Train the model."""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Set special tokens if they don't exist
    special_tokens = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
    }

    # Add special tokens if they don't exist
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        eval_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        report_to="none",
    )

    # Preprocess datasets
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train dataset",
    )
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing validation dataset",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return trainer


def main():
    # Add torch device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    OUTPUT_DIR = "hebrew_qa_model"
    TRAIN_TEST_SPLIT = 0.2

    print("Loading dataset...")
    try:
        dataset = HebrewQADataset("../../Hebrew-Question-Answering-Dataset/data/train.json")
        full_dataset = dataset.to_huggingface_dataset()
        print(f"Dataset loaded successfully with {len(full_dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    print("Splitting dataset...")
    train_test = full_dataset.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=42)
    print(f"Train size: {len(train_test['train'])}, Test size: {len(train_test['test'])}")

    print("Starting training...")
    try:
        trainer = train(
            model_name=MODEL_NAME,
            train_dataset=train_test['train'],
            eval_dataset=train_test['test'],
            output_dir=OUTPUT_DIR
        )

        print("\nEvaluating model...")
        eval_results = trainer.evaluate()
        print("\nEvaluation Results:")
        print(eval_results)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()