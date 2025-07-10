# Numerical-Simulations-of-Rayleigh-Taylor-Driven-Convection-in-Magnetic-White-Dwarfs
This repository contains the code and numerical simulation results accompanying the study:  
**"Numerical Simulations of Rayleigh-Taylor Driven Convection in Magnetic White Dwarfs"**

The project models the interplay between phase separation and fluid motion in white dwarfs using a coupled Cahn-Hilliard–Navier-Stokes framework. Simulations were carried out using the [Dedalus](https://dedalus-project.readthedocs.io/) spectral PDE solver.

## Overview

White dwarfs — the compact remnants of low- to intermediate-mass stars — often possess strong magnetic fields, the origins of which are still under investigation. This project explores the hypothesis that a **convective dynamo**, driven by **Rayleigh-Taylor instability** during crystallization, could sustain magnetic fields within white dwarfs.

The simulations model:
- **Phase separation** using the **Cahn-Hilliard equation**
- **Fluid dynamics** using the **anelastic Navier-Stokes equations**
- A **coupled system (CHNS)** that integrates these phenomena

Two main simulation modes are included:
- `decoupled_simulation/`: Cahn-Hilliard-only phase separation (no fluid motion)
- `coupled_simulation/`: Fully coupled CHNS simulation with buoyancy and advection

---

## 📁 Repository Structure

```bash
.
├── decoupled_simulation/        # Phase separation only
│   ├── dimensionless_decoupled.py             # Main script for decoupled CH simulation
├── coupled_simulation/          # Fully coupled CHNS model
│   ├── dimensionless.py              # Main script for coupled simulation
├── plots/                       # Selected 3D phase and velocity field visualizations
├── white_dwarfs.pdf             # Paper containing full methodology and results
└── README.md                    # You're here
