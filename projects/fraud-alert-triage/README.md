# Fraud Alert Triage Optimization

A binary optimization model for deciding which fraud alerts should receive scarce human-investigation capacity.

A fraud score by itself does not solve the operating problem. Alerts differ in amount at risk, preventability, and investigation time. The optimizer therefore chooses the subset with the highest portfolio-level expected net value.

## Objective

Maximize:

```text
expected prevented loss - analyst review cost
```

## Constraints

- total investigator minutes;
- maximum number of alerts that can be opened.

## Run

```bash
python -m banking_optimization.fraud_triage
pytest tests/test_fraud_triage.py
```

## Extension path

Add investigator skill matching, queue SLAs, sequential evidence gathering, alert dependencies, uncertainty calibration, false-positive externalities, and online re-optimization.

## Limitations

Synthetic educational model only. It is not a fraud decision system and should not be used to block customers, transactions, or accounts.
