import argparse
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("src_file")
parser.add_argument("dst_file")

args = parser.parse_args()

dataset = pandas.read_csv(args.src_file, sep=" ", header=None, names=["word", "pos_tag", "chunk_tag", "ner_tag"], skip_blank_lines=False)

output_dataset = dataset[["word", "ner_tag"]]
output_dataset.to_csv(args.dst_file, sep="\t", index=False, header=False)