# Description

This repository uses an old version of the [Open Applied Topology](https://github.com/pnnl/oat_pnnl) project to implement the U-match PRH algorithm.

## Contributions

The original work in this repository is as follows:

- **jupyter_demo/prh_lag.ipynb** includes helper functions to generate and plot data in order to visualize a few examples of U-match PRH applied to simple and interpretable examples.
- **oat_rust/src/algebra/chains/barcode.rs** implements the function `barcode_relative_homology_lag_filtration`, which computes a PRH barcode and corresponding basis of relative cycle representatives.
- **oat_python/src/clique_filtered.rs** implements the wrapper function `persistent_relative_homology_lag_filtration` that sends our computed results from Rust to Python.

## Build and Run

The typical workfolw will be to create a Pyton virtual environment in which you will compile the backend Rust code that is needed to run the frontend Python and demos in Jupyter. To get started, follow the [directions for building and running OAT](https://github.com/OpenAppliedTopology/oat_python/blob/main/CONTRIBUTING.md). Ensure to follow the optional step to use your own local version of the backend code that is contained within this repository. Once you have completed this, all that is left is to launch the Jupyter notebook contained in the directory `jupyter_demo`. To do this, ensure that you have run `pip install jupyter` while in your active Python virtual environment. Then, CD into the directory **jupyter_demo** and run `jupyter notebook`. Your terminal should provide the local server, and you should select your virtual environment as your kernel.
