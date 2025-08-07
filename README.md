# Project Title

## Short Project Description
This project builds upon the work of Lars Lorch et al. (2024) on **Causal Modeling with Stationary Diffusions**. It implements and extends their approach with minor modifications and additional features aimed at improving [briefly mention your key contribution if possible].

---

## Credits

This project is based on the work:

Lorch, Lars, Andreas Krause, and Bernhard Schölkopf.  
*Causal Modeling with Stationary Diffusions*.  
International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.  
[PMLR Link](https://proceedings.mlr.press/v206/lorch23a.html)

The original implementation can be found here:  
[https://github.com/larslorch/stadion/tree/main](https://github.com/larslorch/stadion/tree/main)

This repository extends and adapts the original codebase by incorporating [briefly mention your key additions or changes].

---

## Related Work

- Master Thesis: [Title or Placeholder]  
  [https://mediatum.ub.tum.de/node?id=1780552](https://mediatum.ub.tum.de/node?id=1780552)

- Upcoming ArXiv Paper: [Title Placeholder]  
  (Will be linked here once available)

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
You can generate data and perform hyperparameter tuning with the following command:

```bash
python hyperparam.py --submit --n_datasets 50 --only_methods ours-linear_u_diag ours-lnl_u_diag --only_gen_types linear-er scm-er
```

Refer to the Lorch Results section in the AISTATS 2024 paper for detailed evaluation and benchmark descriptions.

## Additional Notes
- Add detailed project description and goals here.
- Document additional usage examples or tutorials.
- Include information about expected input/output formats.
- Add references to other related projects or papers.
- Consider adding a FAQ or troubleshooting section.
- Link to your contact info, issue tracker, or contribution guidelines.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## ToDo

- A **detailed explanation** of the methodology or algorithms.
- **Screenshots** or **plots** of key results.
- Information about **file structure** and **data format**.
- **Citation instructions** for your work.
- Links to **video presentations** or demos if any.
- **Contributing guidelines** and **code of conduct** if you expect collaborators.
