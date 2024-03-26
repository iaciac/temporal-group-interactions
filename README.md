# The temporal dynamics of group interactions in higher-order social networks

This repository contains the code required to reproduce the results presented in the following paper:
- I. Iacopini, G. Petri, A. Baronchelli & A. Barrat (2021), [The temporal dynamics of group interactions in higher-order social networks
](https://arxiv.org/abs/2306.09967), ArXiv preprint 2306.09967 (2023).

# Data

The Copenhagen Network Study data are available from the original source~\cite{sapiezynski2019interaction} at \href{https://doi.org/10.6084/m9.figshare.7267433}{https://doi.org/10.6084/m9.figshare.7267433}.
The DyLNet data are available from the original source~\cite{dai2022longitudinal} at \href{https://doi.org/10.7303/syn26560886}{https://doi.org/10.7303/syn26560886}. The GESIS data are available upon request from the original source~\cite{genois2023combining} at \href{https://doi.org/10.7802/2351}{https://doi.org/10.7802/2351}.

 
 This study relies on publicly available datasets that have been collected and released in previous publications:
 
- `CNS` **University** interactions collected by the Copenhagen Network Study, presented in [Sapiezynski et al. 2019](https://www.nature.com/articles/s41597-019-0325-x) and downloaded from [here](https://doi.org/10.6084/m9.figshare.7267433);
- `DyLNet` **Preschool** interactions presented in [Dai et al. 2022](https://www.nature.com/articles/s41597-022-01756-x) and downloaded from [here](https://www.synapse.org/#!Synapse:syn26560886/wiki/616194);
- **Conferences** interactions collected by the [SocioPatterns Collaboration](https://sociopatterns.org/), presented in [Génois et al. 2023](https://doi.org/10.5964/ps.9957) and downloaded ---upon request--- from [here](https://doi.org/10.7802/2351).

Original data are saved into the `data-raw` folder.

# Structure

- `code` contains most of the Python scripts
- `data-raw` contains the sub-folders `CNS`, `DyLNet`, and `Confs` where the original data are stored
- `data-processed` contains the data after the pre-processing described in `data-curation`
- `data-curation` contains the Jupyter notebooks that perform the pre-processing of the raw data from `data-raw`, saving the results in `data-processed`
- `data-analysis` contains the Jupyter notebooks that analyse the empirical data already pre-processed in the `data-processed` folder
- `model` contains the Jupyter notebooks that run the model and analyse the synthetic data
- `Plotting_figures_*.ipynb` are Jupyter notebooks that produce most of the figures
