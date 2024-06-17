from openai import OpenAI
from rdflib import Graph
from rdflib.compare import to_isomorphic, graph_diff
from rdflib.plugins.parsers.notation3 import BadSyntax
from tqdm import tqdm

DATASET = "groups"
BATCH_SIZE = 10
SYSTEM_PROMPT = (
    "The user will provide a set of individuals in CSV format. "
    "Convert all rows of the csv to RDF format, following the examples provided by the user before. All individuals must exist on the response. Convert all elements "
    "without asking the user to take any additional steps.\n"
)

client = OpenAI(api_key="sk-...")

#with open(f"data/{DATASET}.owl", "r") as file_:
#    semantic_model = file_.read()

def batch(iterable, n=1):
    # Source from: https://stackoverflow.com/a/8290508
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


with open(f"data/{DATASET}/data.rdf", "r") as file_:
    rdf_graphs = [example.strip() for example in file_.read().split("############################## Spacer between instances\n")]

with open(f"data/{DATASET}/data.csv", "r") as file_:
    data = file_.readlines()
    header = data[0].strip()
    data = data[1:]


data = list(zip(data, rdf_graphs))

examples = data[:2]#list(zip(data[:2], rdf_graphs[:2]))
data_to_convert = data[2:25]#list(zip(data[2:], rdf_graphs[2:]))

for batch in tqdm(batch(data_to_convert, BATCH_SIZE)):
    input_ = [d[0] for d in batch]
    validation_rdf = [d[1] for d in batch]
    input_ = "\n".join(input_)
    reference_graph = "\n".join(validation_rdf)
    prompt = f"### Input CSV\n{input_}"

    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": examples[0][0]},
            {"role": "assistant", "content": examples[0][1]},
            {"role": "user", "content": examples[1][0]},
            {"role": "assistant", "content": examples[1][1]},
            {"role": "user", "content": prompt},
        ]
    )


    result_graph = chat_completion.choices[0].message.content


    reference = Graph().parse(data=reference_graph, format="turtle")
    try:
        result = Graph().parse(data=result_graph, format="turtle")
    except (BadSyntax, AssertionError) as e:
        print("Bad syntax in the result")
        continue # Skip isomorphic check

    iso_reference = to_isomorphic(reference)
    iso_result = to_isomorphic(result)

    def dump_nt_sorted(g):
        for l in sorted(g.serialize(format='nt').splitlines()):
            if l:
                print(l)

    in_both, in_refence, in_result = graph_diff(iso_reference, iso_result)
    if len(in_refence) > 0:
        print("Only in reference:")
        dump_nt_sorted(in_refence)
    if len(in_result) > 0:
        print("Only in result:")
        dump_nt_sorted(in_result)

    if len(in_refence) == 0 and len(in_result) == 0:
        print("All triples are correct.")
