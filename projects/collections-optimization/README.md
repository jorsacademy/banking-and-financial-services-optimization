# Collections and Recovery Optimization

A mixed-integer resource-allocation model for assigning synthetic delinquent accounts to collection treatments.

The optimizer chooses at most one treatment per account while respecting treatment capacity and an operating budget.

## Treatments

- SMS;
- agent call;
- restructuring workflow;
- legal workflow.

Each account/action pair has a synthetic expected incremental recovery based on balance, delinquency stage, contactability, and action effectiveness.

## Objective

Maximize expected incremental recovery minus treatment cost.

## Constraints

- at most one treatment per account;
- action-specific operational capacities;
- total treatment budget.

## Run

```bash
python -m banking_optimization.collections
pytest tests/test_collections.py
```

## Extension path

Natural extensions include multi-period treatments, treatment sequencing, cure-state transitions, MDPs, contextual bandits, agent scheduling, right-party-contact uncertainty, and capacity sharing across channels.

## Limitations

Synthetic educational model only. It is not a debt-collection policy or customer-treatment recommendation engine. Real implementations require legal, conduct, vulnerability, consent, communication-frequency, fairness, and jurisdiction-specific controls.
