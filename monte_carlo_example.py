# Monte Carlo Simulation for Insurance Claims
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_policies = 1000        # Number of insurance policies
num_simulations = 5000     # Number of Monte Carlo simulations
claim_probability = 0.05   # Probability of a claim occurring (5%)
mean_claim_amount = 5000   # Mean of claim amount in dollars
std_claim_amount = 3000    # Standard deviation of claim amount in dollars

def run_simulation():
    # Simulate whether each policy has a claim (0 or 1)
    claims_occurred = np.random.binomial(1, claim_probability, num_policies)
    
    # Simulate claim amounts (log-normal distribution) for policies with claims
    claim_amounts = np.zeros(num_policies)
    
    # Parameters for log-normal distribution to achieve desired mean and std
    mu = np.log(mean_claim_amount**2 / np.sqrt(std_claim_amount**2 + mean_claim_amount**2))
    sigma = np.sqrt(np.log(1 + std_claim_amount**2 / mean_claim_amount**2))
    
    # Generate claim amounts only for policies with claims
    claim_amounts[claims_occurred == 1] = np.random.lognormal(mu, sigma, sum(claims_occurred))
    
    # Total loss for the portfolio
    total_loss = np.sum(claim_amounts)
    
    # Additional metrics
    num_claims = sum(claims_occurred)
    avg_claim_size = np.mean(claim_amounts[claims_occurred == 1]) if num_claims > 0 else 0
    
    return total_loss, num_claims, avg_claim_size

def main():
    # Run the simulations
    print("Running", num_simulations, "Monte Carlo simulations...")
    results = []
    for _ in tqdm(range(num_simulations)):
        total_loss, num_claims, avg_claim_size = run_simulation()
        results.append({
            'total_loss': total_loss,
            'num_claims': num_claims,
            'avg_claim_size': avg_claim_size
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate key risk metrics
    expected_loss = results_df['total_loss'].mean()
    var_95 = np.percentile(results_df['total_loss'], 95)  # 95% Value at Risk
    var_99 = np.percentile(results_df['total_loss'], 99)  # 99% Value at Risk
    tvar_95 = results_df[results_df['total_loss'] >= var_95]['total_loss'].mean()  # Tail Value at Risk (95%)
    
    # Display results
    print(f"\nInsurance Portfolio Risk Analysis Results:")
    print(f"===================================================")
    print(f"Expected Annual Loss: ${expected_loss:,.2f}")
    print(f"95% Value at Risk (VaR): ${var_95:,.2f}")
    print(f"99% Value at Risk (VaR): ${var_99:,.2f}")
    print(f"95% Tail Value at Risk (TVaR): ${tvar_95:,.2f}")
    print(f"Average Number of Claims: {results_df['num_claims'].mean():.1f}")
    print(f"Average Claim Size: ${results_df['avg_claim_size'].mean():,.2f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Distribution of total losses
    plt.subplot(2, 2, 1)
    sns.histplot(results_df['total_loss'], kde=True)
    plt.axvline(x=expected_loss, color='r', linestyle='--', label=f'Expected Loss: ${expected_loss:,.0f}')
    plt.axvline(x=var_95, color='g', linestyle='--', label=f'95% VaR: ${var_95:,.0f}')
    plt.axvline(x=var_99, color='y', linestyle='--', label=f'99% VaR: ${var_99:,.0f}')
    plt.title('Distribution of Total Portfolio Losses')
    plt.xlabel('Total Loss ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Distribution of number of claims
    plt.subplot(2, 2, 2)
    sns.histplot(results_df['num_claims'], kde=True, discrete=True)
    plt.axvline(x=results_df['num_claims'].mean(), color='r', linestyle='--', 
                label=f'Avg Claims: {results_df["num_claims"].mean():.1f}')
    plt.title('Distribution of Number of Claims')
    plt.xlabel('Number of Claims')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Distribution of average claim size
    plt.subplot(2, 2, 3)
    sns.histplot(results_df['avg_claim_size'], kde=True)
    plt.axvline(x=results_df['avg_claim_size'].mean(), color='r', linestyle='--', 
                label=f'Avg Claim Size: ${results_df["avg_claim_size"].mean():,.0f}')
    plt.title('Distribution of Average Claim Size')
    plt.xlabel('Average Claim Size ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Scatter plot of number of claims vs total loss
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['num_claims'], results_df['total_loss'], alpha=0.5)
    plt.title('Number of Claims vs Total Loss')
    plt.xlabel('Number of Claims')
    plt.ylabel('Total Loss ($)')
    
    plt.tight_layout()
    plt.savefig('monte_carlo_results.png')
    print("\nVisualizations saved to 'monte_carlo_results.png'")
    plt.show()
    
    # Additional analysis: Calculate required capital
    confidence_levels = [0.9, 0.95, 0.99, 0.995, 0.999]
    capital_requirements = {}
    
    for cl in confidence_levels:
        var = np.percentile(results_df['total_loss'], cl * 100)
        tvar = results_df[results_df['total_loss'] >= var]['total_loss'].mean()
        capital_requirements[cl] = {
            'VaR': var,
            'TVaR': tvar,
            'Required Capital (VaR)': var - expected_loss,
            'Required Capital (TVaR)': tvar - expected_loss
        }
    
    capital_df = pd.DataFrame(capital_requirements).T
    print("\nCapital Requirements at Different Confidence Levels:")
    print(capital_df)
    
    # Simple reinsurance analysis
    # Assume a simple excess-of-loss reinsurance with different attachment points
    attachment_points = [100000, 150000, 200000, 250000, 300000]
    reinsurance_results = {}
    
    for point in attachment_points:
        ceded_losses = [max(0, loss - point) for loss in results_df['total_loss']]
        retained_losses = [min(loss, point) for loss in results_df['total_loss']]
        
        reinsurance_results[point] = {
            'Avg Retained Loss': np.mean(retained_losses),
            'Avg Ceded Loss': np.mean(ceded_losses),
            'Max Retained Loss': np.max(retained_losses),
            '95% VaR (Retained)': np.percentile(retained_losses, 95),
            '99% VaR (Retained)': np.percentile(retained_losses, 99),
        }
    
    reinsurance_df = pd.DataFrame(reinsurance_results).T
    print("\nImpact of Different Reinsurance Attachment Points:")
    print(reinsurance_df)

if __name__ == "__main__":
    main()
