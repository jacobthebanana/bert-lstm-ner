# %%
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt   
from matplotlib.colors import LogNorm


parser = argparse.ArgumentParser("Measure accuracy and consistency of IOB tagging.")
parser.add_argument("input_csv_file")
parser.add_argument("output_img")
args = parser.parse_args()

# %%
table = pd.read_csv(args.input_csv_file, sep=" ")

reference_labels = table["ground_truth"]
output_labels_no_transformer = table["predicted_no_transformer"]
output_labels_with_transformer = table["predicted_with_transformer"]

confusion_matrix_no_transformer = confusion_matrix(reference_labels, output_labels_no_transformer)
ax= plt.subplot()
ax.xaxis.tick_top()

sns.heatmap(confusion_matrix_no_transformer, annot=True, fmt='g', ax=ax, norm=LogNorm(), cmap=plt.cm.Blues)  #annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix- Without Transformer')
feature_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'][::-1]
ax.xaxis.set_ticklabels(feature_names) 
ax.yaxis.set_ticklabels(feature_names)

ax.figure.savefig(f"{args.output_img}-no-transformer.png")

ax.figure.clear()

confusion_matrix_with_transformer = confusion_matrix(reference_labels, output_labels_with_transformer)
ax= plt.subplot()
ax.xaxis.tick_top()

sns.heatmap(confusion_matrix_with_transformer, annot=True, fmt='g', ax=ax, norm=LogNorm(), cmap=plt.cm.Blues)  #annot=True to annotate cells, ftm='g' to disable scientific notation
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix- With Transformer')
feature_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'] 
ax.xaxis.set_ticklabels(feature_names) 
ax.yaxis.set_ticklabels(feature_names)

ax.figure.savefig(f"{args.output_img}-with-transformer.png")

# %%
