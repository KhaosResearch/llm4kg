from pathlib import Path
from statistics import mean

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


dataset = load_dataset("wikitext", 'wikitext-103-raw-v1')
print([i for i in dataset])
data = [
        i["text"] for i in dataset["train"]
        if i["text"].strip() != ""
    ] + \
    [
        i["text"] for i in dataset["validation"]
        if i["text"].strip() != ""
    ] + \
    [
        i["text"] for i in dataset["test"]
        if i["text"].strip() != ""
    ]
print(len(data))
print(mean([len(i) for i in data]))


db_data = []
p = Path("data/").glob('**/*.rdf')
db_files = [x for x in p if x.is_file()]
for file in db_files:
    with open(file, "r") as file_:
        db_data += [
            line for line in file_.readlines()
            if "############################## Spacer between instances" not in line
        ]
print(len(db_data))
print(mean([len(i) for i in db_data]))



tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = Whitespace()

tokenizer.train_from_iterator(data[:169479] + db_data, trainer)
tokenizer.save("tokenizer-wiki.json")


# Evaluate on training set
# rdf_data = []
# p = Path("data/").glob('**/*.rdf')
# rdf_files = [x for x in p if x.is_file()]
# for file in rdf_files:
#     with open(file, "r") as file_:
#         rdf_data += [
#             line for line in file_.readlines()
#             if "############################## Spacer between instances" not in line
#         ]

# Evaluate on MOODY
rdf_data = []

with open("GECCO19.n3", "r") as file_:
    rdf_data += [
        line for line in file_.readlines()
        if "############################## Spacer between instances" not in line
    ]


total = 0
output = tokenizer.encode_batch(rdf_data)


for i in output:
    total += len(i)


print("Total tokens",total)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

total = 0
#output = tokenizer.encode(rdf_data)


for i in rdf_data:
    total += len(tokenizer.encode(i))


print("Total with mistral tokens", total)