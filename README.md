# multimodal-survival-analysis

Code to reproduce experiments and results from the paper "Multi-modal clustering reveals event-free patient subgroup in colorectal cancer survival". The code provides functionality for easy multi-modal data processing including standardised filtering and imputation, training autoencoders to retrieve embeddings, functionality for mapping dna and mirna probes to gene targets, and has scripts that perform gene set enrichment analysis on top k cluster contributing features. 

Data and results associated with this work is hosted on zenodo at https://zenodo.org/records/14604885.

## Development setup

To install the package, from the root folder run:

```console
poetry install
```
## Train embeddings

A single layer autoencoder example:

```console
train-embeddings --model_name AE_1layer --data_path ./multimodal/data/pancan_omics --parameters_file ./multimodal/pancan_embeddings/params_rnaseq.json --output_path ./multimodal/pancan_embeddings
```
## Survival analysis

```console
train-survival --data_path ./multimodal/data/pancan_omics --train_filename merged_wsi_omics.csv --parameters_file ./multimodal/pancan_embeddings/survival_params.json --output_path ./multimodal/wsi_omics_survival --target_name DSS --model_name coxph --model_parameters_grid {'alpha':[0.1,0.5,1]} 
```
Similarly, to run GSEA or compute feature importance, use the ```gsea-analysis``` and ```feature-importance``` commands with the appropriate command line arguments.

## Citation
```
@article{janakarajan2024multiclustersurvival,
  title={Multi-modal clustering reveals event-free patient subgroup in colorectal cancer survival},
  author={Janakarajan, Nikita and Larghero, Guillaume and Martinez, Maria Rodriguez},
  year={2024},
}
```
## IBM Public Repository Disclosure

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.
