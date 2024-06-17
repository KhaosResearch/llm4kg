from pathlib import Path

import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


def finetune(
    dataset: Path,
    base_model: str = "mistralai/Mistral-7B-v0.1",
    output: Path | None = None,
):
    """Finetune a transformer based model based on from download_graph

    It requires as inputs the base model to finetune (from a Hugging Face id)
    and the path of the output of the dataset.

    This function is a high level wrapper of the finetuning process.
    """
    if output is None:
        output = Path("./trained_model/")

    dataset = generate_dataset(dataset)

    model, tokenizer = load_model(base_model)

    dataset = format_prompt_and_tokenize(dataset, tokenizer)
    train_dataset, validate_dataset = dataset.train_test_split(test_size=0.15).values()


    model = apply_lora(model)

    __finetune(model, tokenizer, train_dataset, validate_dataset, output)


def generate_dataset(folder: Path) -> Dataset:
    """Generates a HF Dataset from the dataset"""
    data = []
    for file_ in folder.iterdir():
        if not file_.is_dir():  # Ignore files
            continue
        local_folder = file_
        with open(local_folder / "data.csv", "r") as f:
            csv = f.read()

        with open(local_folder / "data.rdf", "r") as f:
            rdf = f.read()

        header, *csv_rows = csv.split("\n")

        rdf_graphs = rdf.split("############################## Spacer between instances\n")
        

        for csv_row, rdf_graph in zip(csv_rows, rdf_graphs):
            data.append({"header": header, "input": csv_row, "output": rdf_graph})

    df = pd.DataFrame(data)

    return Dataset.from_pandas(df)


def load_model(hf_model_id: str):
    # TODO add option for quantization
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id, quantization_config=None, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_prompt_and_tokenize(dataset: Dataset, tokenizer):
    # TODO this function can be vastly improve
    def formatting_func(example):
        text = (
            f"### Convert the following csv file into an rdf graph:\n"
            f"The csv header is:\n{example['header']}\n"
            f"The csv data is:\n{example['input']}\n"
            f"### rdf graph generated from the data:\n{example['output']}\n"
            "### End"
        )
        return text

    def generate_and_tokenize_prompt(prompt):
        return tokenizer(formatting_func(prompt))

    # TODO: Ideally a fixed size should be used
    # def truncate_and_tokenize(prompt):
    #     result = tokenizer(
    #         formatting_func(prompt),
    #         truncation=True,
    #         max_length=512, # TODO as parameter
    #         padding="max_length",
    #     )
    #     result["labels"] = result["input_ids"].copy()
    #     return result

    return dataset.map(generate_and_tokenize_prompt)


def apply_lora(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    return model


def __finetune(
    model,
    tokenizer,
    tokenized_train_dataset: Dataset,
    tokenized_val_dataset: Dataset,
    output_dir: Path,
):
    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            max_steps=500,
            learning_rate=2.5e-5,  # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=50,  # When to start reporting loss
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=50,  # Save checkpoints every 50 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=50,  # Evaluate and save checkpoints every 50 steps
            do_eval=True,  # Perform evaluation at the end of training
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()


finetune(Path("data/"), "mistralai/Mistral-7B-v0.1")