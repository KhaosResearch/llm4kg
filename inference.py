from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer



def inference(header: str, csv_input: str, checkpoint: str | Path):
    model, tokenizer = load_model(checkpoint)

    internal_prompt = (
            "### Convert the following csv file into an rdf graph:\n"
            f"The csv header is:\n{header}\n"
            f"The csv data is:\n{csv_input}\n"
            f"### rdf graph generated from the data:\n"
        )

    model_input = tokenizer(internal_prompt, return_tensors="pt").to("cuda")

    # Set model in evluation mode
    model.eval()

    # Avoid gradient calculation for less memory usage
    with torch.no_grad():
        model_output_tok = model.generate(
            **model_input, max_new_tokens=2000, repetition_penalty=1.15
        )
        output = tokenizer.decode(model_output_tok[0], skip_special_tokens=True)

    return output.removeprefix(internal_prompt).split("### End")[0]


def load_model(checkpoint: str | Path):
    config = PeftConfig.from_pretrained(checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, checkpoint)

    return model, tokenizer


if __name__ == "__main__":
    with open("validation/books/data.csv", "r") as f:
        csv = f.readlines()

        header = csv[0].strip()
        # Select input line to generate the output
        csv_input = csv[4].strip()

    checkpoint = "trained_model/checkpoint-500"
    print(inference(header, csv_input, checkpoint))