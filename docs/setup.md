---
icon: lucide/settings
---

# Environment Setup

Instructions for setting up the software environment on USC CARC systems.

!!! warning "Prerequisites"

    Make sure you are connected to USC secure wireless or VPN before proceeding.

## Login to CARC OnDemand

1. Go to [https://ondemand.carc.usc.edu/](https://ondemand.carc.usc.edu/)
2. Request a **Discovery Cluster Shell Access** session within Open OnDemand.

## Request an Interactive Session

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=1:00:00
```

For bootcamp reservations (if applicable):

```bash
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32GB --time=1:00:00 --reservation=bootcamp --account=hpcsuppt_613
```

## Install Packages with Conda

```bash
module purge
module load conda
git clone https://github.com/uschpc/Building-NeuralNetworks.git
cd Building-NeuralNetworks
python carc_conda_setup.py
```

### For older GPU nodes (e.g., V100)

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

!!! note

    The `carc_conda_setup.py` script automatically installs common packages such as PyTorch, Matplotlib, Pandas, and others into the Conda environment.
