# %%
import argparse
import pandas as pd
from datasets import load_metric

parser = argparse.ArgumentParser("Get summary statistics for a given output csv.")
parser.add_argument("csv_path")
args = parser.parse_args()

metric = load_metric("seqeval")

# %%
table = pd.read_csv(args.csv_path, sep=" ", skip_blank_lines=False)

reference_labels = table["ground_truth"]
output_labels_no_transformer = table["predicted_no_transformer"]
output_labels_with_transformer = table["predicted_with_transformer"]

# %%
def get_iob_metrics(ground_truth, predictions):
    split_ground_truth = []
    split_predictions = []

    current_sentence_ground_truth = []
    current_sentence_predictions = []

    for ground_truth, prediction in zip(ground_truth, predictions):
        if str(ground_truth) == "nan":
            split_ground_truth.append(current_sentence_ground_truth)
            current_sentence_ground_truth = []

            split_predictions.append(current_sentence_predictions)
            current_sentence_predictions = []

        else:
            current_sentence_ground_truth.append(ground_truth)
            current_sentence_predictions.append(prediction)

    return metric.compute(references=split_ground_truth, predictions=split_predictions)

# %%
metrics = get_iob_metrics(reference_labels.iloc, output_labels_no_transformer.iloc)

row = []
for key in ["overall_accuracy", "overall_f1"]:
    row.append(metrics[key] * 100)

for key in ["LOC", "MISC", "ORG", "PER"]:
    row.append(metrics[key]["f1"] * 100)

print("no-transformer & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% ".format(*row))

# %%
metrics.keys()

# %%
metrics = get_iob_metrics(reference_labels.iloc, output_labels_with_transformer.iloc)

row = []
for key in ["overall_accuracy", "overall_f1"]:
    row.append(metrics[key] * 100)

for key in ["LOC", "MISC", "ORG", "PER"]:
    row.append(metrics[key]["f1"] * 100)

print("with-transformer & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% & {:.2f}\\% ".format(*row))


