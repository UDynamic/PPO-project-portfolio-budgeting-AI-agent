#!/usr/bin/env python3
"""
Generate illustrative charts for the RL Portfolio Budgeting paper.
Run from project root: python article/figures/generate_charts.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import beta, lognorm, truncnorm
import seaborn as sns

# Change to figures directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 10

def generate_scurve_comparison():
    """Generate S-curve cashflow profiles for different project categories."""
    t = np.linspace(0, 1, 100)
    
    # Parameters from the paper
    profiles = {
        'Domestic Low-Risk': (2.00, 1.80),
        'Domestic High-Risk': (1.60, 2.16),
        'International Low-Risk': (2.50, 2.16),
        'International High-Risk': (2.00, 2.59)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, (alpha, beta_param) in profiles.items():
        cumulative = beta.cdf(t, alpha, beta_param)
        ax.plot(t, cumulative, label=label, linewidth=2)
    
    ax.set_xlabel('Normalized Project Duration', fontsize=12)
    ax.set_ylabel('Cumulative Cashflow (%)', fontsize=12)
    ax.set_title('S-Curve Cashflow Profiles by Project Category', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('scurve_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('scurve_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated S-curve comparison chart")

def generate_profit_margin_distributions():
    """Generate profit margin distributions for domestic vs international."""
    # Parameters from the paper
    domestic_mean, domestic_std = 0.10, 0.012
    intl_mean, intl_std = 0.14, 0.030
    
    # Truncated normal distributions
    domestic_a, domestic_b = (0.08 - domestic_mean) / domestic_std, (0.12 - domestic_mean) / domestic_std
    intl_a, intl_b = (0.12 - intl_mean) / intl_std, (0.16 - intl_mean) / intl_std
    
    x_domestic = np.linspace(0.08, 0.12, 200)
    x_intl = np.linspace(0.12, 0.16, 200)
    
    y_domestic = truncnorm.pdf(x_domestic, domestic_a, domestic_b, loc=domestic_mean, scale=domestic_std)
    y_intl = truncnorm.pdf(x_intl, intl_a, intl_b, loc=intl_mean, scale=intl_std)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(x_domestic, y_domestic, alpha=0.6, label='Domestic (8-12%)', color='steelblue')
    ax.fill_between(x_intl, y_intl, alpha=0.6, label='International (12-16%)', color='coral')
    
    ax.axvline(0.10, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.8, label='Domestic Mean (10%)')
    ax.axvline(0.14, color='coral', linestyle='--', linewidth=1.5, alpha=0.8, label='International Mean (14%)')
    
    ax.set_xlabel('Profit Margin', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Profit Margin Distributions by Project Category', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis as percentages
    ax.set_xticks(np.arange(0.08, 0.17, 0.01))
    ax.set_xticklabels([f'{x:.0%}' for x in np.arange(0.08, 0.17, 0.01)])
    
    plt.tight_layout()
    plt.savefig('profit_margins.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('profit_margins.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated profit margin distributions")

def generate_bac_distribution():
    """Generate BAC (Budget at Completion) lognormal distribution."""
    mu_log, sigma_log = 5.01, 0.69
    
    x = np.linspace(50, 1000, 500)
    y = lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(x, y, alpha=0.6, color='mediumseagreen')
    ax.axvline(150, color='darkgreen', linestyle='--', linewidth=2, label='Median ($150M)')
    ax.axvline(280, color='darkred', linestyle='--', linewidth=2, label='Mean ($280M)')
    
    ax.set_xlabel('Budget at Completion (Million USD)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Project BAC Distribution (Lognormal)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(50, 1000)
    
    plt.tight_layout()
    plt.savefig('bac_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('bac_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated BAC distribution")

def generate_spi_cpi_distributions():
    """Generate SPI and CPI distributions."""
    # SPI: Beta(2.8, 2.2) rescaled to [0.3, 1.3]
    # CPI: Beta(3.2, 1.8) rescaled to [0.5, 1.2]
    
    x_spi = np.linspace(0.3, 1.3, 200)
    x_cpi = np.linspace(0.5, 1.2, 200)
    
    # Transform to [0,1] for beta distribution
    u_spi = (x_spi - 0.3) / (1.3 - 0.3)
    u_cpi = (x_cpi - 0.5) / (1.2 - 0.5)
    
    y_spi = beta.pdf(u_spi, 2.8, 2.2) / (1.3 - 0.3)
    y_cpi = beta.pdf(u_cpi, 3.2, 1.8) / (1.2 - 0.5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # SPI
    ax1.fill_between(x_spi, y_spi, alpha=0.6, color='royalblue')
    ax1.axvline(0.86, color='darkblue', linestyle='--', linewidth=2, label='Mean (0.86)')
    ax1.axvline(1.0, color='red', linestyle=':', linewidth=1.5, label='On Schedule (1.0)')
    ax1.set_xlabel('Schedule Performance Index (SPI)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('SPI Distribution at Completion', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # CPI
    ax2.fill_between(x_cpi, y_cpi, alpha=0.6, color='darkorange')
    ax2.axvline(0.87, color='darkred', linestyle='--', linewidth=2, label='Mean (0.87)')
    ax2.axvline(1.0, color='red', linestyle=':', linewidth=1.5, label='On Budget (1.0)')
    ax2.set_xlabel('Cost Performance Index (CPI)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('CPI Distribution at Completion', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spi_cpi_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('spi_cpi_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated SPI/CPI distributions")

def generate_rolling_horizon_diagram():
    """Generate rolling horizon architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Timeline
    total_horizon = 36
    window_size = 12
    
    # Draw three rolling windows
    windows = [0, 6, 12]
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for i, (start, color) in enumerate(zip(windows, colors)):
        # Window rectangle
        rect = plt.Rectangle((start, i*0.3), window_size, 0.2, 
                             facecolor=color, alpha=0.5, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Decision point
        ax.plot(start, i*0.3 + 0.1, 'o', color=color, markersize=12, zorder=10)
        ax.text(start, i*0.3 + 0.35, f't={start}', ha='center', fontsize=10, fontweight='bold')
        
        # Executed action
        if i > 0:
            ax.arrow(windows[i-1], (i-1)*0.3 + 0.1, 0.8, 0, 
                    head_width=0.08, head_length=0.5, fc='black', ec='black', linewidth=2)
    
    # Full timeline
    ax.plot([0, total_horizon], [-0.5, -0.5], 'k-', linewidth=2)
    ax.plot([0, 0], [-0.55, -0.45], 'k-', linewidth=2)
    ax.plot([total_horizon, total_horizon], [-0.55, -0.45], 'k-', linewidth=2)
    ax.text(total_horizon/2, -0.7, 'Full Portfolio Duration (36 months)', ha='center', fontsize=11)
    
    ax.set_xlim(-2, total_horizon + 2)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title('Rolling Horizon Architecture (H=12 months)', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='steelblue', alpha=0.5, label='Planning Window 1'),
        plt.Rectangle((0, 0), 1, 1, fc='coral', alpha=0.5, label='Planning Window 2'),
        plt.Rectangle((0, 0), 1, 1, fc='mediumseagreen', alpha=0.5, label='Planning Window 3'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Decision Point'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Executed Action')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('rolling_horizon.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('rolling_horizon.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated rolling horizon diagram")

def generate_milestone_payment_structure():
    """Generate milestone payment structure visualization."""
    milestones = ['Advance\n(0%)', 'M1\n(25%)', 'M2\n(50%)', 'M3\n(75%)', 'Final\n(100%)']
    payments = [10, 20, 30, 30, 10]
    progress = [0, 25, 50, 75, 100]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Payment structure
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(milestones)))
    bars = ax1.bar(milestones, payments, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Payment Weight (%)', fontsize=12)
    ax1.set_title('Milestone Payment Structure', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 35)
    
    # Add value labels on bars
    for bar, payment in zip(bars, payments):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{payment}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Cumulative payment curve
    cumulative = np.cumsum([0] + payments)
    ax2.plot(progress, cumulative, 'o-', color='darkgreen', linewidth=3, markersize=10)
    ax2.fill_between(progress, cumulative, alpha=0.3, color='green')
    ax2.set_xlabel('Project Progress (%)', fontsize=12)
    ax2.set_ylabel('Cumulative Revenue (%)', fontsize=12)
    ax2.set_title('Cumulative Revenue Collection', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 105)
    
    # Add annotations
    for x, y in zip(progress, cumulative):
        ax2.annotate(f'{y}%', xy=(x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('milestone_payments.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('milestone_payments.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated milestone payment structure")

def generate_portfolio_composition():
    """Generate portfolio composition pie chart."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sizes = [60, 40]
    labels = ['Domestic Projects\n(60%)', 'International Projects\n(40%)']
    colors = ['steelblue', 'coral']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.0f%%', startangle=90, textprops={'fontsize': 13})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title('Optimal Portfolio Composition\n(Risk-Adjusted Return Maximization)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('portfolio_composition.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('portfolio_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated portfolio composition chart")

def main():
    """Generate all charts."""
    print("\nGenerating illustrative charts for the paper...\n")
    print(f"Working directory: {os.getcwd()}\n")
    
    try:
        generate_scurve_comparison()
        generate_profit_margin_distributions()
        generate_bac_distribution()
        generate_spi_cpi_distributions()
        generate_rolling_horizon_diagram()
        generate_milestone_payment_structure()
        generate_portfolio_composition()
        
        print("\n✓ All charts generated successfully!")
        print(f"  Output: {os.getcwd()}/*.pdf and *.png\n")
    except Exception as e:
        print(f"\n✗ Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
