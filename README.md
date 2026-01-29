# Stein-Type Causal Modeling with Stationary Diffusions

## Short Project Description
This project builds upon the work of Lars Lorch et al. (2024) on **Causal Modeling with Stationary Diffusions**. It implements and extends their approach with minor modifications and additional features.
The main attribution is the newly proposed loss function, called 'Stein-Type Kernel Deviation from Stationarity' (SKDS). The theory has been developed during the work on my Master Thesis 'Marginal Independence in Causal Modeling with Stationary Diffusions' [mediatum.ub.tum.de](https://mediatum.ub.tum.de/node?id=1780552)

---

## Credits

This project is based on the work:

Lorch, Krause and Schölkopf.  
*Causal Modeling with Stationary Diffusions*.  
International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.  
[PMLR Link](https://proceedings.mlr.press/v238/lorch24a/lorch24a.pdf)

The original implementation can be found here:  
[https://github.com/larslorch/stadion](https://github.com/larslorch/stadion/tree/main)

This repository extends and adapts the original codebase by incorporating the SKDS.

---

## Installation

To install the dependencies and set up the environment, please follow these steps:

1. Create a new conda environment from the provided `environment.yml`:

   ```bash
   conda env create --file environment.yml
   
2. Activate the environment:

   ```bash
   conda activate stadion

3. Install the 'stadion' package and the adapted 'cdt-source' package:

   ```bash
   pip install -e .
   pip install -e ./cdt-source/

4. If you require the baselines implemented in R, first install R on your system, then install the necessary R packages via:

    ```bash
    Rscript r_install.R

## Usage & Results
For the reproduction of the results, please follow the tutorial at [Lorch / stadion / Results](https://github.com/larslorch/stadion/tree/aistats?tab=readme-ov-file#results)

### Alternatively
For the ease of use you can generate data and perform hyperparameter tuning with the following command:

```bash
python hyperparam.py --submit --n_datasets 50 --only_methods ours-linear_u_diag ours-lnl_u_diag --only_gen_types linear-er scm-er
```

Apply the methods to the generated data with the respective hyperparameters with the following commands:
```bash
python manager.py <experiment-name> --methods --submit
python manager.py <experiment-name> --summary --submit
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Reference

The paper is published on arxiv.org: [Efficient Learning of Stationary Diffusions with Stein-type Discrepancies](https://arxiv.org/abs/2601.16597) 

