عالیه! حالا هر دو فایل رو دارم. 

می‌خوام `sensitivityAnalysis.md` رو به `evaluation.md` ادغام کنم و یک سند یکپارچه بسازم. استراتژی ادغام:

**رویکرد ادغام:**
1. **Executive Summary** رو به‌روز می‌کنم تا Sensitivity Analysis رو هم پوشش بده
2. بخش **Sensitivity Analysis** رو به‌عنوان یک ماژول جدید (در کنار Stress Testing) اضافه می‌کنم
3. **Implementation Code** رو با کدهای جدید (experiment runner، statistical tests، visualization) گسترش می‌دم
4. **Experimental Protocol** رو با پروتکل‌های sensitivity analysis تکمیل می‌کنم
5. **References** رو با منابع مرتبط با robustness testing تکمیل می‌کنم

شروع می‌کنم:
عالیه! حالا هر دو فایل رو دارم. 

می‌خوام `sensitivityAnalysis.md` رو به `evaluation.md` ادغام کنم و یک سند یکپارچه بسازم. استراتژی ادغام:

**رویکرد ادغام:**
1. **Executive Summary** رو به‌روز می‌کنم تا Sensitivity Analysis رو هم پوشش بده
2. بخش **Sensitivity Analysis** رو به‌عنوان یک ماژول جدید (در کنار Stress Testing) اضافه می‌کنم
3. **Implementation Code** رو با کدهای جدید (experiment runner، statistical tests، visualization) گسترش می‌دم
4. **Experimental Protocol** رو با پروتکل‌های sensitivity analysis تکمیل می‌کنم
5. **References** رو با منابع مرتبط با robustness testing تکمیل می‌کنم

شروع می‌کنم:
متوجه شدم که ابزارهای فایل در دسترس نیستند. بذار محتوای یکپارچه رو مستقیم اینجا ارائه بدم:

---

# **Comprehensive Evaluation Framework for RL-Based Project Portfolio Optimization**

## **Executive Summary**

This document presents a rigorous, publication-ready evaluation framework for Reinforcement Learning (RL) agents managing project portfolios under uncertainty. The framework addresses three critical evaluation dimensions:

1. **Optimality Assessment**: Perfect Foresight (PF) baseline + Hybrid Pre-Sampling for fair policy comparison
2. **Robustness Testing**: Stress testing under adversarial scenarios (contractor collapse, intervention resistance, sunk cost traps)
3. **Sensitivity & Stability Analysis**: Systematic testing of hyperparameters, state representations, distribution shifts, and computational efficiency

**Key Contributions:**
- Theoretical upper bound via Perfect Foresight for optimality gap calculation
- Hybrid pre-sampling framework for handling endogenous stochasticity
- Three adversarial stress-test scenarios with statistical protocols
- Comprehensive sensitivity analysis covering 6 critical dimensions
- Complete Python implementation with statistical testing and visualization
- Publication-ready reporting templates and experimental protocols

---

## **Table of Contents**

1. [Core Evaluation Metrics](#1-core-evaluation-metrics)
2. [Perfect Foresight Baseline](#2-perfect-foresight-baseline)
3. [Hybrid Pre-Sampling Framework](#3-hybrid-pre-sampling-framework)
4. [Stress Testing & Failure Mode Analysis](#4-stress-testing--failure-mode-analysis)
5. [Sensitivity & Robustness Analysis](#5-sensitivity--robustness-analysis)
6. [Implementation Code](#6-implementation-code)
7. [Experimental Protocol](#7-experimental-protocol)
8. [Reporting & Visualization](#8-reporting--visualization)
9. [Limitations & Future Work](#9-limitations--future-work)
10. [References](#10-references)

---

## **1. Core Evaluation Metrics**

### **1.1 Primary Performance Metrics**

```python
# Portfolio-level metrics
total_portfolio_value = sum(project.realized_value for project in completed_projects)
completion_rate = len(completed_projects) / len(total_projects)
average_delay = mean(project.completion_time - project.planned_duration)

# Efficiency metrics
resource_utilization = sum(resources_used) / sum(resources_available)
intervention_frequency = count(interventions) / total_timesteps
cost_overrun_rate = sum(actual_costs - budgeted_costs) / sum(budgeted_costs)
```

### **1.2 Risk-Adjusted Metrics**

$$\text{Sharpe Ratio} = \frac{E[\text{Portfolio Value}] - \text{Risk-Free Value}}{\sigma[\text{Portfolio Value}]}$$

$$\text{Value at Risk (VaR)}_{95} = \text{5th percentile of portfolio value distribution}$$

---

## **2. Perfect Foresight Baseline**

### **2.1 Theoretical Foundation**

The Perfect Foresight (PF) agent observes the complete realization of all stochastic processes before making decisions, providing a **theoretical upper bound** on achievable performance.

**Mathematical Formulation:**

$$V^{PF}(s_0) = \max_{\pi} \mathbb{E}_{\omega \sim \Omega}\left[\sum_{t=0}^{T} \gamma^t r_t \mid s_0, \omega\right]$$

where $\omega$ represents the complete trajectory of stochastic realizations.

### **2.2 Implementation**

```python
class PerfectForesightAgent:
    def __init__(self, env):
        self.env = env
        
    def solve(self, scenario_seed):
        """Solve with full knowledge of future stochastic realizations."""
        # Pre-generate all random events
        np.random.seed(scenario_seed)
        future_events = self._generate_all_events()
        
        # Solve deterministic optimization problem
        return self._optimize_with_foresight(future_events)
    
    def _generate_all_events(self):
        """Generate complete trajectory of stochastic events."""
        events = {
            'contractor_performance': np.random.beta(2, 2, size=(n_projects, T)),
            'technical_risks': np.random.binomial(1, p_risk, size=(n_projects, T)),
            'resource_availability': np.random.gamma(shape, scale, size=T)
        }
        return events
    
    def _optimize_with_foresight(self, events):
        """Solve deterministic MIP with known future."""
        model = gp.Model("PF_Portfolio")
        
        # Decision variables
        x = model.addVars(n_projects, T, vtype=GRB.BINARY, name="start")
        y = model.addVars(n_projects, T, vtype=GRB.BINARY, name="continue")
        
        # Objective: maximize NPV with known outcomes
        model.setObjective(
            gp.quicksum(
                project_value[i] * discount_factor[t] * 
                (1 - events['technical_risks'][i, t]) * 
                events['contractor_performance'][i, t]
                for i in range(n_projects) for t in range(T)
            ),
            GRB.MAXIMIZE
        )
        
        # Resource constraints with known availability
        for t in range(T):
            model.addConstr(
                gp.quicksum(resource_usage[i] * y[i, t] for i in range(n_projects))
                <= events['resource_availability'][t]
            )
        
        model.optimize()
        return model.objVal, model.getVars()
```

### **2.3 Optimality Gap Calculation**

$$\text{Optimality Gap} = \frac{V^{PF} - V^{\pi}}{V^{PF}} \times 100\%$$

**Interpretation:**
- Gap < 5%: Near-optimal performance
- Gap 5-15%: Good performance with room for improvement
- Gap > 15%: Significant suboptimality, investigate failure modes

---

## **3. Hybrid Pre-Sampling Framework**

### **3.1 Problem: Endogenous Stochasticity**

Standard RL evaluation faces a critical challenge: **policy decisions affect future stochastic realizations**.

**Example:** Intervening in a failing project changes its future risk profile, making direct comparison between "intervene" and "don't intervene" policies impossible on the same trajectory.

### **3.2 Solution: Hybrid Pre-Sampling**

**Core Idea:** Pre-sample exogenous randomness, allow endogenous responses to differ.

```python
class HybridPreSamplingEvaluator:
    def __init__(self, env, n_scenarios=100):
        self.env = env
        self.n_scenarios = n_scenarios
        self.exogenous_samples = self._presample_exogenous()
    
    def _presample_exogenous(self):
        """Pre-sample policy-independent randomness."""
        samples = []
        for seed in range(self.n_scenarios):
            np.random.seed(seed)
            samples.append({
                'initial_contractor_quality': np.random.beta(2, 2, n_projects),
                'market_conditions': np.random.normal(1.0, 0.1, T),
                'external_shocks': np.random.poisson(0.1, T)
            })
        return samples
    
    def evaluate_policy(self, policy):
        """Evaluate policy on pre-sampled scenarios."""
        results = []
        for scenario in self.exogenous_samples:
            # Reset environment with pre-sampled exogenous factors
            state = self.env.reset(exogenous_state=scenario)
            
            episode_return = 0
            done = False
            
            while not done:
                action = policy.select_action(state)
                
                # Endogenous randomness (e.g., intervention outcomes) 
                # is generated fresh based on current state
                next_state, reward, done, info = self.env.step(
                    action, 
                    exogenous_state=scenario
                )
                
                episode_return += reward
                state = next_state
            
            results.append(episode_return)
        
        return {
            'mean': np.mean(results),
            'std': np.std(results),
            'ci_95': stats.t.interval(0.95, len(results)-1, 
                                      loc=np.mean(results), 
                                      scale=stats.sem(results))
        }
```

### **3.3 Statistical Comparison Protocol**

```python
def compare_policies(policy_A, policy_B, evaluator, alpha=0.05):
    """Statistically compare two policies on same scenarios."""
    results_A = evaluator.evaluate_policy(policy_A)
    results_B = evaluator.evaluate_policy(policy_B)
    
    # Paired t-test (same scenarios for both policies)
    t_stat, p_value = stats.ttest_rel(results_A['returns'], 
                                       results_B['returns'])
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((results_A['std']**2 + results_B['std']**2) / 2)
    cohens_d = (results_A['mean'] - results_B['mean']) / pooled_std
    
    return {
        'mean_difference': results_A['mean'] - results_B['mean'],
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < alpha,
        'interpretation': _interpret_effect_size(cohens_d)
    }

def _interpret_effect_size(d):
    if abs(d) < 0.2: return "negligible"
    elif abs(d) < 0.5: return "small"
    elif abs(d) < 0.8: return "medium"
    else: return "large"
```

---

## **4. Stress Testing & Failure Mode Analysis**

### **4.1 Motivation**

Real-world portfolios face extreme scenarios that standard evaluation may miss. Stress testing reveals:
- **Tail risk exposure**: Performance in worst-case scenarios
- **Policy brittleness**: Sensitivity to distributional shifts
- **Failure modes**: Specific conditions causing catastrophic performance

### **4.2 Adversarial Scenario Design**

#### **Scenario 1: Contractor Collapse**

**Description:** Multiple contractors simultaneously fail, forcing emergency reallocation.

```python
class ContractorCollapseScenario:
    def __init__(self, collapse_rate=0.3, collapse_timestep=20):
        self.collapse_rate = collapse_rate
        self.collapse_timestep = collapse_timestep
    
    def apply(self, env):
        """Inject contractor failures at specified timestep."""
        if env.current_timestep == self.collapse_timestep:
            n_failures = int(self.collapse_rate * env.n_contractors)
            failed_contractors = np.random.choice(
                env.n_contractors, 
                size=n_failures, 
                replace=False
            )
            
            for contractor_id in failed_contractors:
                # Mark contractor as failed
                env.contractors[contractor_id].status = 'FAILED'
                
                # Reassign projects or terminate
                affected_projects = env.get_projects_by_contractor(contractor_id)
                for project in affected_projects:
                    project.contractor = None
                    project.progress_penalty = 0.5  # 50% progress loss
                    
        return env
```

**Expected Behavior:**
- **Robust Policy:** Quickly reallocates resources, prioritizes high-value projects
- **Brittle Policy:** Fails to adapt, continues with failed contractors, misses deadlines

#### **Scenario 2: Intervention Resistance**

**Description:** Interventions have reduced effectiveness, testing over-reliance on corrective actions.

```python
class InterventionResistanceScenario:
    def __init__(self, effectiveness_multiplier=0.3):
        self.effectiveness_multiplier = effectiveness_multiplier
    
    def apply(self, env):
        """Reduce intervention effectiveness globally."""
        original_intervene = env.intervene
        
        def weakened_intervene(project_id, intervention_type):
            result = original_intervene(project_id, intervention_type)
            result['risk_reduction'] *= self.effectiveness_multiplier
            result['progress_boost'] *= self.effectiveness_multiplier
            return result
        
        env.intervene = weakened_intervene
        return env
```

**Expected Behavior:**
- **Robust Policy:** Reduces intervention frequency, focuses on prevention
- **Brittle Policy:** Continues high intervention rate, wastes resources

#### **Scenario 3: Sunk Cost Trap**

**Description:** Early high-investment projects face technical failures, testing termination discipline.

```python
class SunkCostTrapScenario:
    def __init__(self, trap_projects=3, failure_timestep=15):
        self.trap_projects = trap_projects
        self.failure_timestep = failure_timestep
    
    def apply(self, env):
        """Create projects with high sunk costs but low future value."""
        # Select high-investment projects
        high_investment_projects = sorted(
            env.projects, 
            key=lambda p: p.cumulative_cost, 
            reverse=True
        )[:self.trap_projects]
        
        if env.current_timestep == self.failure_timestep:
            for project in high_investment_projects:
                # Reveal low remaining value
                project.remaining_value *= 0.2
                # Increase completion cost
                project.remaining_cost *= 2.0
                
        return env
```

**Expected Behavior:**
- **Robust Policy:** Terminates low-value projects despite sunk costs
- **Brittle Policy:** Continues investing due to sunk cost fallacy

### **4.3 Stress Test Execution Protocol**

```python
class StressTestSuite:
    def __init__(self, base_env, policy, n_runs=50):
        self.base_env = base_env
        self.policy = policy
        self.n_runs = n_runs
        
        self.scenarios = [
            ContractorCollapseScenario(),
            InterventionResistanceScenario(),
            SunkCostTrapScenario()
        ]
    
    def run_all_tests(self):
        """Execute all stress tests and compile results."""
        results = {}
        
        # Baseline performance
        baseline = self._evaluate_baseline()
        results['baseline'] = baseline
        
        # Stress test performance
        for scenario in self.scenarios:
            scenario_results = self._evaluate_scenario(scenario)
            results[scenario.__class__.__name__] = scenario_results
            
            # Calculate performance degradation
            degradation = (baseline['mean'] - scenario_results['mean']) / baseline['mean']
            results[f"{scenario.__class__.__name__}_degradation"] = degradation
        
        return results
    
    def _evaluate_scenario(self, scenario):
        """Evaluate policy under specific stress scenario."""
        returns = []
        
        for run in range(self.n_runs):
            env = copy.deepcopy(self.base_env)
            env = scenario.apply(env)
            
            state = env.reset()
            episode_return = 0
            done = False
            
            while not done:
                action = self.policy.select_action(state)
                state, reward, done, info = env.step(action)
                episode_return += reward
            
            returns.append(episode_return)
        
        return {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'var_95': np.percentile(returns, 5)
        }
```

### **4.4 Failure Mode Classification**

```python
def classify_failure_mode(episode_data):
    """Identify specific failure patterns from episode trajectory."""
    failures = []
    
    # Check for resource thrashing
    if episode_data['intervention_frequency'] > 0.5:
        failures.append({
            'type': 'RESOURCE_THRASHING',
            'severity': 'HIGH',
            'description': 'Excessive interventions without strategic focus'
        })
    
    # Check for sunk cost fallacy
    terminated_projects = [p for p in episode_data['projects'] if p['terminated']]
    high_sunk_cost_terminations = [
        p for p in terminated_projects 
        if p['sunk_cost'] > 0.7 * p['total_budget']
    ]
    if len(high_sunk_cost_terminations) < 0.1 * len(terminated_projects):
        failures.append({
            'type': 'SUNK_COST_FALLACY',
            'severity': 'MEDIUM',
            'description': 'Failure to terminate high-sunk-cost, low-value projects'
        })
    
    # Check for contractor over-reliance
    contractor_concentration = max(episode_data['projects_per_contractor'].values())
    if contractor_concentration > 0.5 * len(episode_data['projects']):
        failures.append({
            'type': 'CONTRACTOR_CONCENTRATION',
            'severity': 'HIGH',
            'description': 'Over-reliance on single contractor creates systemic risk'
        })
    
    return failures
```

---

## **5. Sensitivity & Robustness Analysis**

### **5.1 Overview**

Comprehensive sensitivity analysis ensures that RL policies are:
- **Stable** across hyperparameter variations
- **Robust** to distribution shifts
- **Generalizable** to new scenarios
- **Computationally efficient** for deployment

### **5.2 Hyperparameter Sensitivity**

#### **5.2.1 Critical Hyperparameters**

```python
HYPERPARAMETER_GRID = {
    'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    'discount_factor': [0.95, 0.97, 0.99, 0.995],
    'entropy_coefficient': [0.0, 0.01, 0.05, 0.1],
    'gae_lambda': [0.9, 0.95, 0.98, 1.0],
    'clip_range': [0.1, 0.2, 0.3],
    'batch_size': [64, 128, 256, 512],
    'n_epochs': [3, 5, 10, 20]
}
```

#### **5.2.2 Sensitivity Testing Protocol**

```python
class HyperparameterSensitivityAnalyzer:
    def __init__(self, base_config, env, n_seeds=5):
        self.base_config = base_config
        self.env = env
        self.n_seeds = n_seeds
    
    def analyze_parameter(self, param_name, param_values):
        """Analyze sensitivity to single hyperparameter."""
        results = []
        
        for value in param_values:
            config = self.base_config.copy()
            config[param_name] = value
            
            # Train with multiple seeds
            seed_results = []
            for seed in range(self.n_seeds):
                set_seed(seed)
                agent = train_agent(self.env, config)
                performance = evaluate_agent(agent, self.env, n_episodes=50)
                seed_results.append(performance)
            
            results.append({
                'value': value,
                'mean_performance': np.mean(seed_results),
                'std_performance': np.std(seed_results),
                'min_performance': np.min(seed_results),
                'max_performance': np.max(seed_results)
            })
        
        return self._compute_sensitivity_metrics(results)
    
    def _compute_sensitivity_metrics(self, results):
        """Compute sensitivity statistics."""
        performances = [r['mean_performance'] for r in results]
        
        return {
            'range': max(performances) - min(performances),
            'coefficient_of_variation': np.std(performances) / np.mean(performances),
            'max_degradation': (max(performances) - min(performances)) / max(performances),
            'results': results
        }
```

#### **5.2.3 Visualization**

```python
def plot_hyperparameter_sensitivity(sensitivity_results, param_name):
    """Visualize hyperparameter sensitivity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    values = [r['value'] for r in sensitivity_results['results']]
    means = [r['mean_performance'] for r in sensitivity_results['results']]
    stds = [r['std_performance'] for r in sensitivity_results['results']]
    
    # Performance vs hyperparameter value
    ax1.errorbar(values, means, yerr=stds, marker='o', capsize=5)
    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Mean Performance')
    ax1.set_title(f'Sensitivity to {param_name}')
    ax1.grid(True, alpha=0.3)
    
    # Stability (std) vs hyperparameter value
    ax2.plot(values, stds, marker='s', color='red')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Performance Std Dev')
    ax2.set_title(f'Stability across {param_name}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### **5.3 State Representation Ablation**

#### **5.3.1 Feature Importance Analysis**

```python
class StateRepresentationAblation:
    def __init__(self, base_agent, env, feature_groups):
        self.base_agent = base_agent
        self.env = env
        self.feature_groups = feature_groups
    
    def run_ablation_study(self):
        """Remove feature groups and measure performance impact."""
        baseline_performance = evaluate_agent(self.base_agent, self.env)
        
        results = {'baseline': baseline_performance}
        
        for group_name, feature_indices in self.feature_groups.items():
            # Create ablated environment
            ablated_env = self._create_ablated_env(feature_indices)
            
            # Retrain agent without features
            ablated_agent = train_agent(ablated_env, self.base_agent.config)
            ablated_performance = evaluate_agent(ablated_agent, ablated_env)
            
            # Calculate importance
            importance = (baseline_performance - ablated_performance) / baseline_performance
            
            results[group_name] = {
                'performance': ablated_performance,
                'importance': importance,
                'degradation_pct': importance * 100
            }
        
        return results
    
    def _create_ablated_env(self, ablated_indices):
        """Create environment with specified features masked."""
        class AblatedEnv(self.env.__class__):
            def _get_observation(self):
                obs = super()._get_observation()
                obs[ablated_indices] = 0  # Mask features
                return obs
        
        return AblatedEnv()
```

**Example Feature Groups:**

```python
FEATURE_GROUPS = {
    'project_progress': [0, 1, 2, 3],
    'risk_indicators': [4, 5, 6],
    'resource_utilization': [7, 8, 9],
    'contractor_performance': [10, 11, 12],
    'temporal_features': [13, 14, 15],
    'portfolio_metrics': [16, 17, 18]
}
```

### **5.4 Transfer Learning & Data Efficiency**

#### **5.4.1 Learning Curve Analysis**

```python
def analyze_data_efficiency(env, config, data_fractions=[0.1, 0.25, 0.5, 0.75, 1.0]):
    """Measure performance vs training data size."""
    results = []
    
    full_timesteps = config['total_timesteps']
    
    for fraction in data_fractions:
        config['total_timesteps'] = int(full_timesteps * fraction)
        
        # Train with multiple seeds
        performances = []
        for seed in range(5):
            set_seed(seed)
            agent = train_agent(env, config)
            perf = evaluate_agent(agent, env, n_episodes=50)
            performances.append(perf)
        
        results.append({
            'data_fraction': fraction,
            'timesteps': config['total_timesteps'],
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances)
        })
    
    return results

def plot_learning_curve(data_efficiency_results):
    """Plot performance vs training data."""
    fractions = [r['data_fraction'] for r in data_efficiency_results]
    means = [r['mean_performance'] for r in data_efficiency_results]
    stds = [r['std_performance'] for r in data_efficiency_results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(fractions, means, yerr=stds, marker='o', capsize=5)
    plt.xlabel('Training Data Fraction')
    plt.ylabel('Evaluation Performance')
    plt.title('Data Efficiency: Performance vs Training Data Size')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    return plt.gcf()
```

#### **5.4.2 Transfer Learning Protocol**

```python
class TransferLearningEvaluator:
    def __init__(self, source_env, target_envs):
        self.source_env = source_env
        self.target_envs = target_envs
    
    def evaluate_transfer(self, config):
        """Evaluate transfer learning performance."""
        # Train on source environment
        source_agent = train_agent(self.source_env, config)
        source_performance = evaluate_agent(source_agent, self.source_env)
        
        results = {'source_performance': source_performance}
        
        for target_name, target_env in self.target_envs.items():
            # Zero-shot transfer (no fine-tuning)
            zero_shot_perf = evaluate_agent(source_agent, target_env)
            
            # Fine-tuned transfer
            finetuned_agent = finetune_agent(
                source_agent, 
                target_env, 
                timesteps=config['total_timesteps'] // 10
            )
            finetuned_perf = evaluate_agent(finetuned_agent, target_env)
            
            # Train from scr
atch baseline
            scratch_agent = train_agent(target_env, config)
            scratch_perf = evaluate_agent(scratch_agent, target_env)
            
            results[target_name] = {
                'zero_shot': zero_shot_perf,
                'finetuned': finetuned_perf,
                'from_scratch': scratch_perf,
                'transfer_gain': finetuned_perf - scratch_perf
            }
        
        return results

### **5.5 Robustness to Distribution Shift**

#### **5.5.1 Distribution Shift Categories**

python
DISTRIBUTION_SHIFTS = {
    'increased_risk': {
        'technical_failure_rate': 1.5,
        'contractor_variance': 2.0
    },
    'resource_scarcity': {
        'resource_availability': 0.7,
        'budget_constraints': 0.8
    },
    'market_volatility': {
        'market_noise_std': 2.0
    },
    'project_complexity': {
        'dependency_density': 1.5,
        'task_duration_variance': 1.8
    }
}

#### **5.5.2 Robustness Evaluation**

python
class DistributionShiftEvaluator:
    def __init__(self, trained_agent, base_env):
        self.agent = trained_agent
        self.base_env = base_env
    
    def evaluate_robustness(self, shifts):
        """Test policy under distribution shifts."""
        baseline_perf = evaluate_agent(self.agent, self.base_env)
        
        robustness_results = {}
        
        for shift_name, shift_params in shifts.items():
            shifted_env = self._create_shifted_env(shift_params)
            shifted_perf = evaluate_agent(self.agent, shifted_env)
            
            robustness_score = shifted_perf / baseline_perf
            
            robustness_results[shift_name] = {
                'performance': shifted_perf,
                'robustness_score': robustness_score,
                'degradation_pct': (1 - robustness_score) * 100
            }
        
        return robustness_results
    
    def _create_shifted_env(self, shift_params):
        """Modify environment parameters to create distribution shift."""
        shifted_env = copy.deepcopy(self.base_env)
        
        for param, multiplier in shift_params.items():
            current_value = getattr(shifted_env, param)
            setattr(shifted_env, param, current_value * multiplier)
        
        return shifted_env

### **5.6 Random Seed Stability**

python
class SeedStabilityAnalyzer:
    def __init__(self, env, config, n_seeds=20):
        self.env = env
        self.config = config
        self.n_seeds = n_seeds
    
    def analyze(self):
        """Measure stability across random seeds."""
        performances = []
        training_curves = []
        
        for seed in range(self.n_seeds):
            set_seed(seed)
            
            # Train agent
            agent, training_history = train_agent_with_history(
                self.env, 
                self.config
            )
            
            # Evaluate
            performance = evaluate_agent(agent, self.env)
            
            performances.append(performance)
            training_curves.append(training_history)
        
        return {
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'coefficient_of_variation': np.std(performances) / np.mean(performances),
            'worst_case': np.min(performances),
            'best_case': np.max(performances),
            'iqr': np.percentile(performances, 75) - np.percentile(performances, 25),
            'all_performances': performances,
            'training_curves': training_curves
        }

**Stability Interpretation:**

| Coefficient of Variation | Stability Assessment |
|---|---|
| < 5% | Excellent |
| 5–10% | Good |
| 10–20% | Moderate |
| > 20% | Poor |

### **5.7 Computational Efficiency**

python
class ComputationalEfficiencyAnalyzer:
    def __init__(self, env, configs):
        self.env = env
        self.configs = configs
    
    def benchmark(self):
        """Benchmark computational requirements."""
        results = []
        
        for config_name, config in self.configs.items():
            # Training benchmark
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            agent = train_agent(self.env, config)
            
            training_time = time.time() - start_time
            peak_memory = psutil.Process().memory_info().rss - start_memory
            
            # Inference benchmark
            inference_times = []
            state = self.env.reset()
            
            for _ in range(1000):
                start_inf = time.time()
                _ = agent.select_action(state)
                inference_times.append(time.time() - start_inf)
            
            results.append({
                'config': config_name,
                'training_time_hours': training_time / 3600,
                'peak_memory_gb': peak_memory / 1e9,
                'mean_inference_ms': np.mean(inference_times) * 1000,
                'p95_inference_ms': np.percentile(inference_times, 95) * 1000
            })
        
        return pd.DataFrame(results)

---

# **6. Implementation Code**

## **6.1 Unified Evaluation Runner**

python
class ComprehensiveEvaluationRunner:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
    
    def run_complete_evaluation(self):
        """Execute full evaluation pipeline."""
        
        results = {}
        
        print("Running baseline evaluation...")
        results['baseline'] = self._baseline_evaluation()
        
        print("Running Perfect Foresight comparison...")
        results['optimality'] = self._optimality_analysis()
        
        print("Running stress tests...")
        results['stress_tests'] = self._stress_test_analysis()
        
        print("Running sensitivity analysis...")
        results['sensitivity'] = self._sensitivity_analysis()
        
        print("Running robustness analysis...")
        results['robustness'] = self._robustness_analysis()
        
        print("Generating report...")
        self._generate_report(results)
        
        return results
    
    def _baseline_evaluation(self):
        return evaluate_agent(self.agent, self.env, n_episodes=100)
    
    def _optimality_analysis(self):
        pf_agent = PerfectForesightAgent(self.env)
        pf_value, _ = pf_agent.solve(scenario_seed=42)
        
        rl_value = evaluate_agent(self.agent, self.env)
        
        return {
            'pf_value': pf_value,
            'rl_value': rl_value,
            'optimality_gap_pct': (pf_value - rl_value) / pf_value * 100
        }
    
    def _stress_test_analysis(self):
        suite = StressTestSuite(self.env, self.agent)
        return suite.run_all_tests()
    
    def _sensitivity_analysis(self):
        analyzer = HyperparameterSensitivityAnalyzer(
            self.config,
            self.env
        )
        
        return analyzer.analyze_parameter(
            'learning_rate',
            [1e-5, 1e-4, 1e-3]
        )
    
    def _robustness_analysis(self):
        evaluator = DistributionShiftEvaluator(
            self.agent,
            self.env
        )
        
        return evaluator.evaluate_robustness(
            DISTRIBUTION_SHIFTS
        )

---

# **7. Experimental Protocol**

## **7.1 Reproducibility Requirements**

python
REPRODUCIBILITY_CONFIG = {
    'random_seeds': list(range(20)),
    'evaluation_episodes': 100,
    'confidence_level': 0.95,
    'significance_threshold': 0.05,
    'effect_size_reporting': True,
    'hardware_logging': True,
    'environment_versioning': True
}

## **7.2 Statistical Reporting Standards**

Every reported result must include:
- Mean performance
- Standard deviation
- 95% confidence interval
- Number of seeds/runs
- Statistical significance tests
- Effect size measures

### **Example Reporting Format**

text
PPO achieved a portfolio value of 142.3 ± 8.7 
(95% CI: [138.1, 146.5], n=20 seeds), 
significantly outperforming DQN 
(127.4 ± 11.2, p<0.01, Cohen's d=1.47).

---

# **8. Reporting & Visualization**

## **8.1 Recommended Figures**

1. Learning curves with confidence intervals
2. Optimality gap comparison
3. Stress test degradation heatmap
4. Hyperparameter sensitivity plots
5. Distribution shift robustness radar chart
6. Seed stability violin plots
7. Computational efficiency Pareto frontier

## **8.2 LaTeX Table Template**

latex
\begin{table}[h]
\centering
\caption{Robustness Evaluation Results}
\begin{tabular}{lcccc}
\toprule
Scenario & Mean Return & Std Dev & Degradation (\%) & Robustness Score \\
\midrule
Baseline & 142.3 & 8.7 & 0.0 & 1.00 \\
Contractor Collapse & 118.5 & 14.2 & 16.7 & 0.83 \\
Intervention Resistance & 125.1 & 11.8 & 12.1 & 0.88 \\
Sunk Cost Trap & 131.7 & 10.4 & 7.4 & 0.93 \\
\bottomrule
\end{tabular}
\end{table}

---

# **9. Limitations & Future Work**

## **9.1 Current Scope Limitations**

This framework intentionally excludes:

- Explicit project interdependencies
- Dynamic budget shocks
- Multi-agent contractor competition
- Macroeconomic market volatility
- Human-in-the-loop decision systems
- Regulatory/political interventions

## **9.2 Future Research Directions**

1. Graph neural networks for project dependency modeling
2. Meta-RL for rapid adaptation to distribution shifts
3. Multi-objective RL balancing value, risk, and fairness
4. Causal RL for intervention effect estimation
5. Offline RL using historical portfolio datasets
6. Safe RL with hard operational constraints

---

# **10. References**

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
4. Henderson, P., et al. (2018). Deep RL that Matters.
5. Agarwal, R., et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice.
6. Dulac-Arnold, G., et al. (2021). Challenges of Real-World Reinforcement Learning.
7. Kirk, D. B. (1995). Optimal Control Theory.
8. Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering.
9. Taleb, N. N. (2007). The Black Swan.
10. Goodfellow, I., et al. (2015). Explaining and Harnessing Adversarial Examples.
11. Cobbe, K., et al. (2019). Quantifying Generalization in Reinforcement Learning.
12. Packer, C., et al. (2018). Assessing Generalization in Deep Reinforcement Learning.
13. Mania, H., et al. (2018). Simple random search provides a competitive approach to reinforcement learning.
14. Islam, R., et al. (2017). Reproducibility of Benchmark Results in Deep Reinforcement Learning.
15. Nagarajan, P., et al. (2018). The Impact of Nondeterminism on Reproducibility in Deep RL.

---

# **Conclusion**

This unified framework provides a rigorous, publication-ready methodology for evaluating RL-based project portfolio optimization systems across four dimensions:

- Optimality
- Robustness
- Stability
- Computational practicality

The integration of:
- Perfect Foresight baselines,
- Hybrid pre-sampling,
- Adversarial stress testing,
- Sensitivity analysis,

creates a comprehensive evaluation ecosystem suitable for both academic research and industrial deployment validation.