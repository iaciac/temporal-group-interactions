# The temporal dynamics of group interactions in higher-order social networks

This repository contains the code required to reproduce the results presented in the following paper:
- I. Iacopini, M. Karsai & A. Barrat (2023), [The temporal dynamics of group interactions in higher-order social networks
](https://arxiv.org/abs/2306.09967), ArXiv preprint 2306.09967 (2023).

# Data
 
 This study relies on publicly available datasets that have been collected and released in previous publications:
 
- `CNS` **University** interactions collected by the Copenhagen Network Study, presented in [Sapiezynski et al. 2019](https://www.nature.com/articles/s41597-019-0325-x) and downloaded from [here](https://doi.org/10.6084/m9.figshare.7267433);
- `DyLNet` **Preschool** interactions presented in [Dai et al. 2022](https://www.nature.com/articles/s41597-022-01756-x) and downloaded from [here](https://www.synapse.org/#!Synapse:syn26560886/wiki/616194);
- **Conferences** interactions collected by the [SocioPatterns Collaboration](https://sociopatterns.org/), presented in [Génois et al. 2023](https://doi.org/10.5964/ps.9957) and downloaded ---upon request--- from [here](https://doi.org/10.7802/2351).

Original data are saved into the `data-raw` folder.

# Structure

- `code` contains most of the Python scripts
- `data-raw` contains the sub-folders `CNS`, `DyLNet`, and `Confs` where the original data are stored
- `data-curation` contains the Jupyter notebooks that perform the pre-processing of the raw data from `data-raw`, saving the results in `data-processed`
- `data-processed` contains the data after the pre-processing described in `data-curation`
- `data-analysis` contains the Jupyter notebooks that analyse the empirical data already pre-processed in the `data-processed` folder
- `model` contains the Jupyter notebooks that run the model and analyse the synthetic data

# Dependencies

The code of the paper strongly relies on a number of exeternal Python libraries. The code has been originally run on a machine containing the following Python dependencies:

- matplotlib 3.5.1
- networkx 2.8
- numpy 1.21.6
- palettable 3.3.0
- pandas 1.5.3
- powerlaw 1.5
- scipy 1.10.0
- xgi 0.5.6
