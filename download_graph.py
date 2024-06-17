from SPARQLWrapper import SPARQLWrapper, JSON
import csv
import re
from pathlib import Path

from query import ALL

def query_dbpedia(query):
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    # Execute the query and fetch the results
    results = sparql.query().convert()
    return results

def export_as_rdf(results, output_file, template):
    fill_values = []
    for result in results["results"]["bindings"]:
        fill_values.append({
            k: v
            for k, v in result.items()
        })
    rdf = fill_template(fill_values, template)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(rdf)

def format_value(v, type, datatype):
    v
    if type == "uri":
        v = f"<{v}>"
    elif type == "typed-literal":
        v = f'"{v}"^^<{datatype}>'
    elif type == "literal":
        v = f'"{v}"'
    else:
        raise ValueError(type, datatype, v)
    return v

def fill_template(values, template):
    rdf = []
    # Regular expression to match OPTIONAL blocks and capture the content inside the braces
    pattern = r'OPTIONAL\s*{\s*(.*?)\s*}\s*\.\s*\n'

    # Use re.sub() to replace OPTIONAL blocks with their content
    template = re.sub(pattern, r'\1 .\n', template)
    template = template.strip()
    for value in values:
        new_template = template
        for key in value:
            # Split the value by semicolon if it contains multiple values
            v_binding = value[key]
            split_values = v_binding["value"].split(';')
            if len(split_values) > 1:
                # Find the line that contains the variable
                lines = new_template.split('\n')
                new_lines = []
                for line in lines:
                    if f"?{key}" in line:
                        # Duplicate the line for each value in split_values
                        for v in split_values:
                            new_lines.append(line.replace(f"?{key}", format_value(v.strip(), v_binding["type"], v_binding.get("datatype", None))))
                    else:
                        new_lines.append(line)
                new_template = '\n'.join(new_lines)
            else:
                # Single value replacement
                new_template = new_template.replace(f"?{key}", v_binding["value"])
        

        # Clean unusued variables
        # Define the regex pattern to match lines containing "?" that defines variables in SPARQL
        pattern = re.compile(r'^.*\?.*$', re.MULTILINE)

        # Use the regex pattern to remove matching lines
        new_template = re.sub(pattern, '', new_template)

        new_template = "\n".join([line for line in new_template.splitlines() if line])
        rdf.append(new_template)
    return "\n############################## Spacer between instances\n".join(rdf)


def export_as_csv(results, output_file):
    vars = results["head"]["vars"]
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=vars)
        writer.writeheader()
        for result in results["results"]["bindings"]:
            row = {var: result[var].get("value", None) for var in vars if var in result}
            writer.writerow(row)

if __name__ == "__main__":
    for k, v in ALL.items():
        results = query_dbpedia(v["query"])

        result_path = Path(f"data/{k}")
        result_path.mkdir(exist_ok=True)

        # Export results as RDF
        export_as_rdf(results, result_path/ f"data.rdf", v["template"])

        # Export results as CSV
        export_as_csv(results, result_path/ f"data.csv")

        print(f"Export of {k} complete.")
