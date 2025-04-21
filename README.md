# Monte Carlo Insurance Risk Simulation

This repository contains tools for setting up and running Monte Carlo simulations for insurance risk analysis in an OpenShift environment.

## Overview

Monte Carlo simulations are computational algorithms that rely on repeated random sampling to obtain numerical results. In insurance risk analysis, they help estimate the probability distribution of outcomes by simulating thousands of possible scenarios.

## Contents

- `monte_carlo_setup.yml`: Ansible playbook to set up a Jupyter notebook environment in OpenShift
- `monte_carlo_example.py`: Example Monte Carlo simulation for insurance risk analysis

## Getting Started

### Prerequisites

- Access to an OpenShift cluster
- Ansible Automation Platform with access to the cluster

### Using the Ansible Playbook

The Ansible playbook will deploy a Jupyter notebook environment in your OpenShift cluster with the following components:

1. A namespace called `insurance-risk`
2. A Jupyter notebook deployment with data science libraries pre-installed
3. A service to expose the Jupyter server
4. A route to make the Jupyter server accessible from outside the cluster

To run the playbook from Ansible Automation Platform:

1. Set up a Project pointing to this repository
2. Create a Job Template using the `monte_carlo_setup.yml` playbook
3. Launch the Job Template

### Example Monte Carlo Simulation

The `monte_carlo_example.py` file provides a complete Monte Carlo simulation for insurance risk analysis. It simulates:

- A portfolio of 1,000 insurance policies
- The probability of claims occurring
- The severity of claims when they do occur
- The aggregate loss distribution

It calculates key risk metrics such as:

- Expected Loss
- Value at Risk (VaR) at different confidence levels
- Tail Value at Risk (TVaR)
- Capital requirements
- Impact of different reinsurance strategies

## Key Risk Metrics Explained

- **Expected Loss**: The average loss amount across all simulations
- **Value at Risk (VaR)**: The loss amount that won't be exceeded with a certain confidence level
- **Tail Value at Risk (TVaR)**: The average of all losses that exceed the VaR threshold
- **Required Capital**: Additional funds needed to cover potential losses beyond the expected loss

## Customizing the Simulation

You can modify the following parameters to match your specific insurance portfolio:

- `num_policies`: Number of insurance policies in the portfolio
- `claim_probability`: Probability of a claim occurring for each policy
- `mean_claim_amount`: Average size of claims
- `std_claim_amount`: Standard deviation of claim sizes

## Using the Jupyter Environment

Once deployed, you can access the Jupyter environment using the URL provided at the end of the playbook execution. From there, you can:

1. Upload the `monte_carlo_example.py` file
2. Create new notebooks to run and modify the simulations
3. Visualize the results with interactive charts
4. Export your findings for reporting

## License

This project is licensed under the MIT License - see the LICENSE file for details.
