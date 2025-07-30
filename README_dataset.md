## Few-Shot Data Selection per task
### Attribute Value Extraction(AVE)
- OA-Mine: from ExtractGPT. [Paper](https://arxiv.org/abs/2310.12537) [Repo](https://github.com/wbsg-uni-mannheim/ExtractGPT)
- Few-shot Selection: we use 20% data from the original repo in `data/processed_datasets/oa-mine/train_0.2.jsonl`, while the full data is `data/processed_datasets/oa-mine/train_1.0.jsonl`. Test data is `data/processed_datasets/oa-mine/test.jsonl`
- Data Augmentation: since original data contains multiple attributes and targed values, e.g. 3 attribute-key values, we extract it as 3 separate instruction-input-output records.
- the processed data is stored in `dataset/AVE`
- the checkpoint data is stored in `lora_weight/Expert/AVE/AVE-oa_mine`
### Data Imputation(DI)
- Restaurant/Walmart/Amazon: from [ER_Magellan benchmarks](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md).
- for dataset `Restaurant`, we merge the left and right table for benchmark `Fodors-Zagats` in ER_Magellan benchmarks, and treat them as one dataset. For dataset `walmart` and `amazon`, we select the left table of `Walmart-Amazon` and the right table of `Amazon-Google` separately. Please check `raw_dataset/DI` for the merged file.
- For the few-shot index for our experiment, we random select: 10% of all data above as few-shot data, 70% of all data as unlabeled data, and the remain 20% data as test. e.g., for dataset `amazon`, please check `raw_dataset/DI/amazon_train_few_index.npy` for the index of labeled file in `raw_dataset/DI/amazon/amazon_all_filter_train.csv`
- Following the setting of baseline paper [IPM](https://github.com/EliasMei/IPM), we set the following attribute as missint:
```
ATTR_DICT = {
    "walmart":["category", "brand"],
    "amazon":["category", "brand"],
    "restaurant":["city"]
}
```
- Data-Augmentation: we run the embedding model $M_{RAG}$ which is fine-tuned over the labeled data, to pseudo-label all the unlabeled data to construct the train file and the in-context demonstration. As result, for dataset `restaurant`, we expand the labeled set from 86 to 566, `amazon` from 2k to 8k, and `walmart` from 242 to 6939.(Such process is not necessary for all dataset, and the file size can be a lot smaller, by filtering representative records via data diversity) Please check the `brand_predict/category_predict` row for the pseudo-label prediction result.
- train/test file: please check `dataset/DI` for the train/test file
- the checkpoint is stored in `lora_weight/Expert/DI`

## Schema Matching

- all data are from paper [SMAT](https://pmc.ncbi.nlm.nih.gov/articles/PMC8487677/pdf/nihms-1722415.pdf) and [repo](https://github.com/JZCS2018/SMAT)
- Since all data for schema matching are highly biased, we do not apply data augmentation and use all file for training. Please check `dataset/SM` for the train/valid file. 
- the checkpoint is stored in `lora_weight/Expert/SM` 

## Relation-Extraction
- all data comes form [TURL_data](https://github.com/sunlab-osu/TURL), the train data is processed from `train.table_rel_extraction.json`, while the test data is processed from `test.table_rel_extraction.json`
- we select 10% of all data as train file. The selection index in stored in `raw_datatset/RE/RE_sample_10_index.npy`
- Similarity, we use $M_{RAG}$ to annotate `top-k` candidate relations and `top-p` most-relative in-context demonstrations, please check `raw_datatset/RE/` for the processed file.
- train/test file: please check `dataset/RE` for the train/test file
- the checkpoint is stored in `lora_weight/Expert/RE` 

## Column Type Annotation
- SimTab and WebTables are from RECA repo.
- We select 20% data of all datasets as training file. For WebTable, we merge `k0,k1,k2,k3` as all train file, and filter 20% of it as few-shot, while we treat `k4` as test; for SimTab, we treat `train_val_hard_jaccard_ranking.jsonl` as train file. All index can be found as `raw_dataset/CTA` correspondingly.
- Similarity, we use $M_{RAG}$ to annotate `top-k` candidate colume type and `top-p` most-relative in-context demonstrations as data augmentation.
- Due to the large size of all data(for single table with multiple column types and rows, we have to treat them as multiple records, with different context), we sample representative context(e.g. subset of each single table) via data diversity. The train/test file is stored in `dataset/CTA`
- the checkpoint is stored in `lora_weight/Expert/CTA`

## Data Cleaning
- All data comes from Baran repo. and the few-shot setting are kept the same with Baran.
- for each dataset, e.g. `hospital`, the sampled index is stored in `raw_dataset/DC/hospital/index.npy`. The clean and dirty data of benchmark is stored in `raw_dataset/DC/hospital/original/clean.csv` and `raw_dataset/DC/hospital/original/dirty.csv`
- Data Augmentation: we use LLM to generate error detection and data cleaning rules via few-shot samples, and use them to augment data. Please check the appendix of this [Paper](https://github.com/SICS-Fundamental-Research-Center/GIDCL/blob/main/supplementary/GIDCL_Revision_v6_appendix.pdf) for the generated rules.
- The test file only correct the error that are detected from the previous Error Detection results. The detection result is stored in `raw_dataset/DC/hospital/detector/detector.npy`, recording all potential error position.
- The train/test file is stored in `dataset/DC`
- the checkpoint is stored in `lora_weight/Expert/DC`

## Entity Matching 
- `Walmart-Amazon`, `Abt-Buy`, `Amazon-Google` dataset are ER benchmark from [ER_Magellan benchmarks](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md). 
- `WDC-All` dataset are benchmar dataset from [WDC Product Corpus](https://webdatacommons.org/largescaleproductcorpus/v2/index.html). We select the small-size dataset. All setting, containing format and train/valid/test split, are kept the same with [ditto](https://github.com/megagonlabs/ditto/tree/master/data/wdc).
- `Semi-text-Watch` and `Semi-Text-Computer` are from repo [PromptEM](https://github.com/ZJU-DAILY/PromptEM).
- For the few-shot setting, please check `raw_dataset/ER/Walmart-Amazon/index.csv` or `index.npy` for our sampled text.
- Data Augmentation: we use offline LLM to extract additional attributes, and replace the original record with generated structural output. The following are LLM-generated attribute per dataset:
```
ATTR_DICT = {
    "Walmart-Amazon":['title', 'category', 'brand', 'modelno', 'price', 'subcategory', 'key_features', 'sku', 'color'],
    "Amazon-Google":['title', 'manufacturer', 'price', 'category', 'subcategory', 'platform', 'edition', 'type', 'modelno'],
    "WDC-All":["title","category","subcategory","brand","modelno","sku","key_features"],
    "Abt-Buy":['name', 'description', 'price', 'category', 'sku', 'brand', 'modelno', 'key_features']
    "Semi-Text-Watch":['title', 'brand', 'color', 'gender', 'sku', 'diameter', 'description'],
    "Semi-Text-Computer":['title', 'category', 'subcategory', 'brand', 'sku', 'type', 'description'],
}
```
We also provide the prompt for the following generation process. Please check `raw_dataset/ER/Walmart-Amazon/enrich_query_walmart_amazon.csv` for an example. We train $M_\text{RAG}$ over few-shot labeled data, to retrieve unlabeled similar pairs for pairwise LLM generation. Please check `raw_dataset/ER/Walmart-Amazon/train.csv` and `test.csv` for the LLM-generated result.
- $M_\text{RAG}$ is also used to generate pseudo-labeled negative pairs. For dataset `Semi-text-Watch` and `Semi-Text-Computer`, we additional retrieve their `sku` as master data, then self-annotate additional positive(same `sku`) and negative data. 
- The train/test file is stored in `dataset/ER`
- the checkpoint is stored in `lora_weight/Expert/ER`