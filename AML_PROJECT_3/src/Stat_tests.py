import numpy as np
from statsmodels.stats.weightstats import ttost_paired
from statsmodels.stats.multitest import multipletests
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def equivalence_test_rff_kde(df_results, D_values, equivalence_margin=0.02, alpha=0.05):
    """
    Two One-Sided Test (TOST) for equivalence testing.
    
    H0: |AUC_exact - AUC_rff| >= equivalence_margin (models differ)
    H1: |AUC_exact - AUC_rff| < equivalence_margin (models equivalent)
    
    Parameters:
    -----------
    equivalence_margin : float
        Practical equivalence threshold (default 0.02 = 2% AUC difference)
    """
    
    results_dict = {}
    
    for difficulty in df_results['Difficulty'].unique():
        print(f"\n{'='*70}")
        print(f"DIFFICULTY: {difficulty}")
        print(f"{'='*70}")
        
        df_diff = df_results[df_results['Difficulty'] == difficulty]
        exact_aucs = df_diff[df_diff['Method'] == 'Exact_KDE']['AUC'].values
        
        test_results = []
        
        for D in D_values:
            rff_aucs = df_diff[df_diff['D'] == D]['AUC'].values
            
            # Use reliable statsmodels function
            pval_tost, _, _ = ttost_paired(
                exact_aucs, 
                rff_aucs, 
                low=-equivalence_margin, 
                upp=equivalence_margin
            )
            
            # Handle tuple return
            if isinstance(pval_tost, tuple):
                pval_tost = pval_tost[0]
            
            differences = exact_aucs - rff_aucs
            mean_diff = np.mean(differences)
            
            test_results.append({
                'D': D,
                'mean_diff': mean_diff,
                'p_tost': pval_tost
            })
        
        # Holm-Bonferroni correction
        pvalues = [r['p_tost'] for r in test_results]
        reject, pvals_corr, _, _ = multipletests(pvalues, alpha=alpha, method='holm')
        
        for i, r in enumerate(test_results):
            r['corrected_p_tost'] = pvals_corr[i]
            r['equivalent'] = reject[i]
        
        min_D = next((r['D'] for r in test_results if r['equivalent']), None)
        results_dict[difficulty] = {'results': test_results, 'min_D': min_D}
        
        # Display
        print(f"\nMargin: ±{equivalence_margin:.3f}")
        print(f"{'D':<8} {'Mean Δ':<10} {'p(equiv)':<12} {'Adj.p':<12} {'Decision'}")
        print("-"*70)
        for r in test_results:
            decision = "EQUIVALENT" if r['equivalent'] else "NOT equiv"
            print(f"{r['D']:<8} {r['mean_diff']:<10.5f} {r['p_tost']:<12.6f} "
                  f"{r['corrected_p_tost']:<12.6f} {decision}")
        
        if min_D:
            print(f"\n✓ Minimum D: {min_D}")
    
    return results_dict

# ============================================================================
# COMPLETE TOST EQUIVALENCE TEST WITH BOOTSTRAP SUPPORT
# Handles both normal and non-normal data automatically
# ============================================================================




def bootstrap_tost_paired(exact_vals, rff_vals, equivalence_margin=0.02, 
                          n_bootstrap=10000, random_state=42):
    """
    Bootstrap TOST for paired data (distribution-free).

    Parameters:
    -----------
    exact_vals, rff_vals : array-like
        Paired values to compare
    equivalence_margin : float
        Equivalence threshold
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple: (p_tost, p_lower, p_upper, ci_90_lower, ci_90_upper)
    """
    np.random.seed(random_state)

    n = len(exact_vals)
    differences = exact_vals - rff_vals
    mean_diff_obs = np.mean(differences)

    # Bootstrap resampling
    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_diff = differences[indices]
        bootstrap_means[i] = np.mean(boot_diff)

    # Calculate 90% bootstrap confidence interval
    ci_90_lower = np.percentile(bootstrap_means, 5)
    ci_90_upper = np.percentile(bootstrap_means, 95)

    # Bootstrap TOST p-values
    # Lower test: H0: mean <= -margin vs H1: mean > -margin
    # Count how many bootstrap means are <= -margin
    p_lower = np.mean(bootstrap_means <= -equivalence_margin)

    # Upper test: H0: mean >= margin vs H1: mean < margin
    # Count how many bootstrap means are >= margin
    p_upper = np.mean(bootstrap_means >= equivalence_margin)

    # TOST p-value is the maximum
    p_tost = max(p_lower, p_upper)

    return p_tost, p_lower, p_upper, ci_90_lower, ci_90_upper


def equivalence_test_rff_kde(df_results, D_values, equivalence_margin=0.02, 
                             alpha=0.05, method='auto', check_assumptions=True, 
                             save_diagnostics=True, n_bootstrap=10000, random_state=42):
    """
    TOST equivalence testing with automatic parametric/bootstrap selection.

    H0: |AUC_exact - AUC_rff| >= equivalence_margin (models differ)
    H1: |AUC_exact - AUC_rff| < equivalence_margin (models equivalent)

    Parameters:
    -----------
    df_results : pd.DataFrame
        Results with columns: Difficulty, Method, D, AUC
    D_values : array-like
        D values to test
    equivalence_margin : float
        Equivalence threshold (default 0.02)
    alpha : float
        Significance level (default 0.05)
    method : str
        'auto': Automatic selection based on normality
        'parametric': Force parametric TOST (assumes normality)
        'bootstrap': Force bootstrap TOST (no assumptions)
    check_assumptions : bool
        Whether to check normality assumption
    save_diagnostics : bool
        Whether to save diagnostic plots
    n_bootstrap : int
        Number of bootstrap samples (if using bootstrap)
    random_state : int
        Random seed for bootstrap

    Returns:
    --------
    results_dict : dict
        Test results for each difficulty level

    Example:
    --------
    >>> df_results = pd.DataFrame(results)
    >>> stratified_results = equivalence_test_rff_kde(
    ...     df_results, D_VALUES, method='auto'
    ... )
    """

    results_dict = {}

    # Validate inputs
    required_cols = ['Difficulty', 'Method', 'D', 'AUC']
    for col in required_cols:
        assert col in df_results.columns, f"Missing '{col}' column"

    assert equivalence_margin > 0, "Equivalence margin must be positive"
    assert 0 < alpha < 1, "Alpha must be between 0 and 1"
    assert method in ['auto', 'parametric', 'bootstrap'], "Invalid method"

    print("\n" + "="*80)
    print(" TOST EQUIVALENCE TESTING: RFF vs EXACT KDE")
    print("="*80)
    print(f"Equivalence margin: Â±{equivalence_margin:.4f} AUC")
    print(f"Significance level: Î± = {alpha}")
    print(f"Method selection: {method}")
    print(f"Multiple testing correction: Holm-Bonferroni")
    print(f"Number of D values: {len(D_values)}")
    if method == 'bootstrap' or method == 'auto':
        print(f"Bootstrap samples: {n_bootstrap}")
    print("="*80)

    for difficulty in sorted(df_results['Difficulty'].unique()):
        print(f"\n{'='*80}")
        print(f"DIFFICULTY: {difficulty.upper()}")
        print(f"{'='*80}")

        df_diff = df_results[df_results['Difficulty'] == difficulty]
        exact_aucs = df_diff[df_diff['Method'] == 'Exact_KDE']['AUC'].values
        n_seeds = len(exact_aucs)

        if n_seeds < 10:
            print(f"âš  WARNING: Only {n_seeds} seeds - low statistical power")

        print(f"Sample size: {n_seeds} seeds per D value\n")

        test_results = []
        normality_violations = []
        methods_used = []

        for D in D_values:
            rff_aucs = df_diff[df_diff['D'] == D]['AUC'].values

            if len(rff_aucs) != n_seeds:
                raise ValueError(f"Mismatched lengths for D={D}")

            # Calculate differences
            differences = exact_aucs - rff_aucs
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            se_diff = std_diff / np.sqrt(n_seeds)

            # Check normality
            if check_assumptions and n_seeds >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(differences)
                is_normal = shapiro_p > 0.05

                if not is_normal:
                    normality_violations.append(D)
                    print(f"  âš  D={D}: Non-normal differences (Shapiro p={shapiro_p:.4f})")
            else:
                shapiro_p = np.nan
                is_normal = True

            # Decide which method to use
            if method == 'auto':
                # Use parametric if normal OR n>=30 (CLT applies)
                if is_normal:
                    selected_method = 'parametric'
                else:
                    selected_method = 'bootstrap'
            else:
                selected_method = method

            methods_used.append(selected_method)

            # Run selected test
            if selected_method == 'parametric':
                # Standard parametric TOST
                tost_result = ttost_paired(
                    exact_aucs, rff_aucs,
                    low=-equivalence_margin,
                    upp=equivalence_margin
                )

                pval_tost = tost_result[0]
                pval_lower_tuple = tost_result[1]
                pval_upper_tuple = tost_result[2]

                if isinstance(pval_tost, tuple):
                    pval_tost = pval_tost[0]

                if isinstance(pval_lower_tuple, tuple):
                    pval_lower = pval_lower_tuple[1]
                else:
                    pval_lower = pval_lower_tuple

                if isinstance(pval_upper_tuple, tuple):
                    pval_upper = pval_upper_tuple[1]
                else:
                    pval_upper = pval_upper_tuple

                # Calculate CIs
                t_crit_90 = stats.t.ppf(0.95, df=n_seeds-1)
                ci_90_lower = mean_diff - t_crit_90 * se_diff
                ci_90_upper = mean_diff + t_crit_90 * se_diff

                t_crit_95 = stats.t.ppf(0.975, df=n_seeds-1)
                ci_95_lower = mean_diff - t_crit_95 * se_diff
                ci_95_upper = mean_diff + t_crit_95 * se_diff

                test_method_label = 'Parametric'

            else:  # Bootstrap
                # Bootstrap TOST
                pval_tost, pval_lower, pval_upper, ci_90_lower, ci_90_upper = \
                    bootstrap_tost_paired(
                        exact_aucs, rff_aucs,
                        equivalence_margin=equivalence_margin,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state
                    )

                # Calculate 95% CI from bootstrap as well
                np.random.seed(random_state)
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    indices = np.random.choice(n_seeds, size=n_seeds, replace=True)
                    boot_diff = differences[indices]
                    bootstrap_means.append(np.mean(boot_diff))

                ci_95_lower = np.percentile(bootstrap_means, 2.5)
                ci_95_upper = np.percentile(bootstrap_means, 97.5)

                test_method_label = 'Bootstrap'

            # Calculate effect size (works for both methods)
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0

            # Store results
            test_results.append({
                'D': D,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'se_diff': se_diff,
                'p_tost': pval_tost,
                'p_lower': pval_lower,
                'p_upper': pval_upper,
                'ci_90_lower': ci_90_lower,
                'ci_90_upper': ci_90_upper,
                'ci_95_lower': ci_95_lower,
                'ci_95_upper': ci_95_upper,
                'cohens_d': cohens_d,
                'shapiro_p': shapiro_p,
                'normality_ok': is_normal,
                'n_seeds': n_seeds,
                'test_method': test_method_label
            })

        # Holm-Bonferroni correction
        pvalues = [r['p_tost'] for r in test_results]
        reject, pvals_corr, alphacSidak, alphacBonf = multipletests(
            pvalues, alpha=alpha, method='holm'
        )

        for i, r in enumerate(test_results):
            r['corrected_p_tost'] = pvals_corr[i]
            r['equivalent'] = reject[i]

        # Find minimum D
        min_D = next((r['D'] for r in test_results if r['equivalent']), None)

        # Store results
        results_dict[difficulty] = {
            'results': test_results,
            'min_D': min_D,
            'normality_violations': normality_violations,
            'methods_used': dict(zip(D_values, methods_used)),
            'n_comparisons': len(D_values),
            'n_seeds': n_seeds
        }

        # Display results
        print(f"{'D':<8} {'Mean Î”':<10} {'SE':<9} {'p(equiv)':<11} {'Adj.p':<11} "
              f"{'90% CI':<24} {'Cohen d':<9} {'Method':<11} {'Decision'}")
        print("-"*110)

        for r in test_results:
            ci_str = f"[{r['ci_90_lower']:>7.5f}, {r['ci_90_upper']:>7.5f}]"
            decision = "EQUIV" if r['equivalent'] else "NOT equiv"
            normality_flag = " norm " if r['normality_ok'] else " not norm"

            print(f"{r['D']:<8} {r['mean_diff']:<10.6f} {r['se_diff']:<9.6f} "
                  f"{r['p_tost']:<11.6f} {r['corrected_p_tost']:<11.6f} "
                  f"{ci_str:<24} {r['cohens_d']:<9.3f} {r['test_method']:<11} "
                  f"{decision}{normality_flag}")

        # Summary
        if min_D:
            print(f"\n{'='*80}")
            print(f"âœ“ RECOMMENDATION: Minimum D = {min_D}")
            print(f"{'='*80}")
            best_result = next(r for r in test_results if r['D'] == min_D)
            print(f"  Mean AUC difference: {best_result['mean_diff']:.6f}")
            print(f"  Standard error:      {best_result['se_diff']:.6f}")
            print(f"  90% CI:              [{best_result['ci_90_lower']:.6f}, "
                  f"{best_result['ci_90_upper']:.6f}]")
            print(f"  95% CI:              [{best_result['ci_95_lower']:.6f}, "
                  f"{best_result['ci_95_upper']:.6f}]")
            print(f"  Effect size (d):     {best_result['cohens_d']:.3f}")
            print(f"  TOST p-value:        {best_result['p_tost']:.6f}")
            print(f"  Adjusted p-value:    {best_result['corrected_p_tost']:.6f}")
            print(f"  Test method:         {best_result['test_method']}")
            print(f"\n  Interpretation: RFF with D={min_D} achieves statistical")
            print(f"                  equivalence within Â±{equivalence_margin:.4f} AUC margin")
        else:
            print(f"\n{'='*80}")
            print(f"NO EQUIVALENCE at margin Â±{equivalence_margin:.4f}")
            print(f"{'='*80}")
            best = min(test_results, key=lambda x: abs(x['mean_diff']))
            print(f"  Closest: D={best['D']}, mean diff={best['mean_diff']:.6f}")
            print(f"  90% CI:  [{best['ci_90_lower']:.6f}, {best['ci_90_upper']:.6f}]")
            print(f"  Adj. p:  {best['corrected_p_tost']:.6f}")
            print(f"\n  Recommendations:")
            print(f"    (1) Test larger D values (current max: {max(D_values)})")
            print(f"    (2) Relax margin (try Î´=0.03 or Î´=0.05)")
            print(f"    (3) Sensitivity analysis")

        # Print method summary
        if method == 'auto':
            n_parametric = sum(1 for m in methods_used if m == 'parametric')
            n_bootstrap = sum(1 for m in methods_used if m == 'bootstrap')
            print(f"\n  Method breakdown: {n_parametric} parametric, {n_bootstrap} bootstrap")

        # Save diagnostics
        if save_diagnostics and check_assumptions:
            _save_diagnostic_plots(df_diff, D_values, exact_aucs, difficulty,
                                   equivalence_margin, test_results)

    return results_dict


def _save_diagnostic_plots(df_diff, D_values, exact_aucs, difficulty, margin, test_results):
    """Create diagnostic plots for assumption checking."""
    n_plots = min(4, len(D_values))
    fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))

    if n_plots == 1:
        axes = axes.reshape(2, 1)

    for idx, D in enumerate(D_values[:n_plots]):
        rff_aucs = df_diff[df_diff['D'] == D]['AUC'].values
        differences = exact_aucs - rff_aucs
        result = next(r for r in test_results if r['D'] == D)

        # Q-Q plot
        stats.probplot(differences, dist="norm", plot=axes[0, idx])
        axes[0, idx].set_title(
            f'Q-Q Plot: D={D}\n'
            f'Shapiro p={result["shapiro_p"]:.4f}\n'
            f'Method: {result["test_method"]}',
            fontsize=10
        )
        axes[0, idx].grid(True, alpha=0.3)

        # Histogram
        axes[1, idx].hist(differences, bins=15, edgecolor='black', 
                         alpha=0.7, color='steelblue')
        axes[1, idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        axes[1, idx].axvline(-margin, color='orange', linestyle='--', 
                            linewidth=2, label=f'Â±{margin}')
        axes[1, idx].axvline(margin, color='orange', linestyle='--', linewidth=2)
        axes[1, idx].axvline(result['mean_diff'], color='green', linestyle='-',
                            linewidth=2, label=f'Mean={result["mean_diff"]:.4f}')

        axes[1, idx].set_xlabel('AUC Difference (Exact - RFF)', fontsize=10)
        axes[1, idx].set_ylabel('Frequency', fontsize=10)
        axes[1, idx].set_title(
            f'Distribution: D={D}\n'
            f'MeanÂ±SE = {result["mean_diff"]:.4f}Â±{result["se_diff"]:.4f}',
            fontsize=10
        )
        axes[1, idx].legend(fontsize=8)
        axes[1, idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filename = f'tost_diagnostics_{difficulty}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n Diagnostic plots saved: {filename}")


def save_results_to_csv(results_dict, filename='tost_results_complete.csv'):
    """Save detailed results to CSV."""
    all_results = []
    for difficulty, res_dict in results_dict.items():
        for r in res_dict['results']:
            all_results.append({
                'Difficulty': difficulty,
                'D': r['D'],
                'mean_diff_AUC': r['mean_diff'],
                'std_diff': r['std_diff'],
                'se_diff': r['se_diff'],
                'p_tost': r['p_tost'],
                'p_lower': r['p_lower'],
                'p_upper': r['p_upper'],
                'corrected_p_tost': r['corrected_p_tost'],
                'ci_90_lower': r['ci_90_lower'],
                'ci_90_upper': r['ci_90_upper'],
                'ci_95_lower': r['ci_95_lower'],
                'ci_95_upper': r['ci_95_upper'],
                'cohens_d': r['cohens_d'],
                'shapiro_p': r['shapiro_p'],
                'normality_ok': r['normality_ok'],
                'test_method': r['test_method'],
                'equivalent': r['equivalent'],
                'n_seeds': r['n_seeds']
            })

    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)
    print(f"\nâœ“ Detailed results saved to '{filename}'")
    return df


def print_summary_table(results_dict):
    """Print summary across all difficulty levels."""
    print("\n" + "="*80)
    print(" SUMMARY: MINIMUM D FOR EQUIVALENCE BY DIFFICULTY")
    print("="*80)

    summary_data = []
    for difficulty, res in results_dict.items():
        min_D = res['min_D']
        if min_D:
            result = next(r for r in res['results'] if r['D'] == min_D)
            summary_data.append({
                'Difficulty': difficulty,
                'Min_D': min_D,
                'Mean_Diff': f"{result['mean_diff']:.6f}",
                '90%_CI': f"[{result['ci_90_lower']:.5f}, {result['ci_90_upper']:.5f}]",
                'Cohens_d': f"{result['cohens_d']:.3f}",
                'Adj_p': f"{result['corrected_p_tost']:.6f}",
                'Method': result['test_method'],
                'Normal': 'normal' if result['normality_ok'] else ' not normal'
            })
        else:
            summary_data.append({
                'Difficulty': difficulty,
                'Min_D': 'None',
                'Mean_Diff': 'N/A',
                '90%_CI': 'N/A',
                'Cohens_d': 'N/A',
                'Adj_p': 'N/A',
                'Method': 'N/A',
                'Normal': 'N/A'
            })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Final recommendation
    valid_min_Ds = [res['min_D'] for res in results_dict.values() if res['min_D']]
    if valid_min_Ds:
        conservative_D = max(valid_min_Ds)
        median_D = int(np.median(valid_min_Ds))
        print(f"\n{'='*80}")
        print(f"  FINAL RECOMMENDATION")
        print(f"{'='*80}")
        print(f"  Conservative (max): D = {conservative_D}")
        print(f"  Moderate (median):  D = {median_D}")
        print(f"\n  Use D = {conservative_D} for equivalence across ALL difficulties.")
    else:
        print(f"\n  No D achieved equivalence at current margin.")

    print("="*80)