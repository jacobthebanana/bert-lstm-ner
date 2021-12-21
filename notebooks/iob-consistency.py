# %%
import pandas as pd
import argparse

parser = argparse.ArgumentParser("Measure accuracy and consistency of IOB tagging.")
parser.add_argument("input_csv_file")
args = parser.parse_args()

# %%
table = pd.read_csv(args.input_csv_file, sep=" ")

reference_labels = table["ground_truth"]
output_labels_no_transformer = table["predicted_no_transformer"]
output_labels_with_transformer = table["predicted_with_transformer"]

# %%
current_entity = None
num_mismatches = 0

for entry in reference_labels:
    if len(entry) >= 3:
        position = entry[0]
        entity = entry[2:]
        
        if position == "B":
            current_entity = entity

        elif position == "I":
            if current_entity != entity:
                num_mismatches += 1 
                current_entity = entity

print(f"Ground Truth: {num_mismatches / len(table) * 100: .5f}% inconsistence tagging.",)

# %%
current_entity = None
num_mismatches = 0

for entry in output_labels_no_transformer:
    if len(entry) >= 3:
        position = entry[0]
        entity = entry[2:]
        
        if position == "B":
            current_entity = entity

        elif position == "I":
            if current_entity != entity:
                num_mismatches += 1 
                current_entity = entity
    
    else:
        current_entity = None

print(f"No Transformer (baseline): {num_mismatches / len(table) * 100: .5f}% inconsistence tagging.",)

# %%
current_entity = None
num_mismatches = 0

for entry in output_labels_with_transformer:
    if len(entry) >= 3:
        position = entry[0]
        entity = entry[2:]
        
        if position == "B":
            current_entity = entity

        elif position == "I":
            if current_entity != entity:
                num_mismatches += 1 
                current_entity = entity

print(f"With Transformer: {num_mismatches / len(table) * 100: .5f}% inconsistence tagging.",)


