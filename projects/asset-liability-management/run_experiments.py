"""Run the stochastic ALM flagship experiment from the repository root."""

from banking_optimization.alm_experiments import export_experiment_outputs


if __name__ == "__main__":
    paths = export_experiment_outputs(
        "projects/asset-liability-management/outputs"
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
