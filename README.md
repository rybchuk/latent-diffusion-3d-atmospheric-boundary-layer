# Latent Diffusion Models for Atmospheric Large Eddy Simulation
[![DOI](https://zenodo.org/badge/704639836.svg)](https://zenodo.org/doi/10.5281/zenodo.10206880)

This repo contains much of the code developed in the manuscript ["Ensemble flow reconstruction in the atmospheric boundary layer from spatially limited measurements through latent diffusion models" (Rybchuk et al. 2023a)](https://doi.org/10.1063/5.0172559) and deployed in ["A baseline for ensemble-based, time-resolved inflow reconstruction for a single turbine using large-eddy simulations and latent diffusion models" (Rybchuk et al. 2023b)](https://www.doi.org/10.1088/1742-6596/2505/1/012018). The code here builds on [the original LDM repo](https://github.com/CompVis/latent-diffusion) and is purpose built for a specific application: given synthetic measurements from an approximation of the observing network in the Rotor Aerodynamics Aeroelastics & Wake (RAAW) field campaign, generate an ensemble of plausible large eddy simulation states that could be used as initial conditions in the AMR-Wind code. Our intent in sharing the code is to make reproducibility easier, as well as provide a demo on how to modify the original LDM code for other researchers who hope to apply this algorithm to their problem.

This codebase works very similarly to the original LDM codebase, although we carry out inpainting differently. Here, inpainting is achieved by running `main.py` through `slurm/inpaint_*` as opposed to using an additional script.

Below, we plug one of our LDM-generated samples back into our LES code, and demonstrate that the LES successfully runs and doesn't crash :) 

https://github.com/rybchuk/latent-diffusion-3d-atmospheric-boundary-layer/assets/8021012/e3330e9d-e781-4b2b-8b74-4d47eef3c6d1

