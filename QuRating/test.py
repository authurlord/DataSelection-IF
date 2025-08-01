from datasets import load_dataset,load_from_disk

ds = load_from_disk("/data/home/wangys/DataSelection-IF/QuRating/QuRating-Data/QuRating-GPT3.5-Judgments")

print(ds[0])
