# Building-NN-Zensical

Zensical documentation site for the CARC Building Neural Networks workshop.

## Build & Preview

```bash
pip install zensical
zensical serve
```

Then open [http://localhost:8000](http://localhost:8000).

## Build for Production

```bash
zensical build
```

The static site is generated in the `site/` directory.

## Deploy to GitHub Pages

This repository includes a GitHub Actions workflow that automatically builds and deploys the site to GitHub Pages on every push to the `main` branch.

### One-time setup

1. Go to the repository **Settings** → **Pages**
2. Under **Build and deployment**, set **Source** to **GitHub Actions**
3. Save the changes

### After enabling Pages

- The workflow (`.github/workflows/docs.yml`) will run automatically on pushes to `main`
- You can also trigger it manually from the **Actions** tab
- The live site will be available at:  
  **https://jihao2021.github.io/Building-NN-Zensical/**

> **Note:** The first deployment may take a minute or two after you enable GitHub Pages.

## Source Materials

Adapted from [uschpc/Building-NeuralNetworks](https://github.com/uschpc/Building-NeuralNetworks).
