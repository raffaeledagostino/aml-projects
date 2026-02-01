import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_rff_comparative_analysis(df, D_VALUES, DIFFICULTIES):
# Create visualizations for each difficulty level
    for DIFFICULTY in DIFFICULTIES:
        print(f"\n{'='*80}")
        print(f"Analysis for {DIFFICULTY.upper()} Difficulty")
        print(f"{'='*80}")
        
        # Filter data for this difficulty
        df_diff = df[df['Difficulty'] == DIFFICULTY]
        df_kde  = df_diff[df_diff['Method'] == 'Exact_KDE']
        df_gnb  = df_diff[df_diff['Method'] == 'Gaussian_NB']
        df_rff  = df_diff[df_diff['Method'] == 'RFF_KDE']
        
        # Create comprehensive visualization with 4 rows
        fig = plt.figure(figsize=(18, 16))
        gs  = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # ---------------------------------------------------------
        # Row 1: AUC Analysis
        # ---------------------------------------------------------
        # 1.1: AUC vs D
        ax1 = fig.add_subplot(gs[0, 0])
        rff_auc_mean = df_rff.groupby('D')['AUC'].mean()
        rff_auc_std  = df_rff.groupby('D')['AUC'].std()
        ax1.errorbar(
            rff_auc_mean.index, rff_auc_mean.values,
            yerr=rff_auc_std.values,
            marker='o', capsize=5, label='RFF-KDE'
        )
        ax1.axhline(
            df_kde['AUC'].mean(), color='red', linestyle='--',
            label='Exact KDE'
        )
        ax1.fill_between(
            D_VALUES,
            df_kde['AUC'].mean() - df_kde['AUC'].std(),
            df_kde['AUC'].mean() + df_kde['AUC'].std(),
            color='red', alpha=0.2
        )
        ax1.axhline(
            df_gnb['AUC'].mean(), color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax1.fill_between(
            D_VALUES,
            df_gnb['AUC'].mean() - df_gnb['AUC'].std(),
            df_gnb['AUC'].mean() + df_gnb['AUC'].std(),
            color='green', alpha=0.2
        )
        ax1.set_xlabel('Number of RFF Components (D)')
        ax1.set_ylabel('AUC')
        ax1.set_title('AUC vs RFF Components')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # 1.2: AUC Error (difference from Exact KDE)
        ax2 = fig.add_subplot(gs[0, 1])
        exact_auc_mean = df_kde['AUC'].mean()
        auc_error      = df_rff.groupby('D')['AUC'].mean() - exact_auc_mean
        auc_error_std  = df_rff.groupby('D')['AUC'].std()
        ax2.errorbar(
            auc_error.index, auc_error.values,
            yerr=auc_error_std.values,
            marker='o', capsize=5, color='green'
        )
        ax2.axhline(
            0, color='red', linestyle='--', label='Exact KDE baseline',
            linewidth=2
        )
        ax2.set_xlabel('Number of RFF Components (D)')
        ax2.set_ylabel('AUC Error (RFF - Exact)')
        ax2.set_title('Approximation Error')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # 1.3: AUC Boxplot
        ax3 = fig.add_subplot(gs[0, 2])
        df_rff_copy = df_rff.copy()
        df_rff_copy['D_str'] = df_rff_copy['D'].astype(str)
        sns.boxplot(
            data=df_rff_copy, x='D_str', y='AUC', ax=ax3, color='skyblue'
        )
        ax3.axhline(
            df_kde['AUC'].mean(), color='red', linestyle='--', label='Exact KDE'
        )
        ax3.axhline(
            df_gnb['AUC'].mean(), color='green', linestyle=':', label='Gaussian NB'
        )
        ax3.set_xlabel('RFF Components (D)')
        ax3.set_ylabel('AUC')
        ax3.set_title('AUC Distribution')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # ---------------------------------------------------------
        # Row 2: Time Analysis
        # ---------------------------------------------------------
        # 2.1: Training Time vs D
        ax4 = fig.add_subplot(gs[1, 0])
        train_mean = df_rff.groupby('D')['Train_Time_sec'].mean()
        train_std  = df_rff.groupby('D')['Train_Time_sec'].std()
        ax4.errorbar(
            train_mean.index, train_mean.values,
            yerr=train_std.values,
            marker='s', capsize=5, label='RFF-KDE', color='blue'
        )
        ax4.axhline(
            df_kde['Train_Time_sec'].mean(), color='red', linestyle='--',
            label='Exact KDE'
        )
        ax4.fill_between(
            D_VALUES,
            df_kde['Train_Time_sec'].mean() - df_kde['Train_Time_sec'].std(),
            df_kde['Train_Time_sec'].mean() + df_kde['Train_Time_sec'].std(),
            color='red', alpha=0.2
        )
        ax4.axhline(
            df_gnb['Train_Time_sec'].mean(), color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax4.fill_between(
            D_VALUES,
            df_gnb['Train_Time_sec'].mean() - df_gnb['Train_Time_sec'].std(),
            df_gnb['Train_Time_sec'].mean() + df_gnb['Train_Time_sec'].std(),
            color='green', alpha=0.2
        )
        ax4.set_xlabel('Number of RFF Components (D)')
        ax4.set_ylabel('Training Time (sec)')
        ax4.set_title('Training Time vs D')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        # 2.2: Inference Time vs D
        ax5 = fig.add_subplot(gs[1, 1])
        infer_mean = df_rff.groupby('D')['Infer_Time_sec'].mean()
        infer_std  = df_rff.groupby('D')['Infer_Time_sec'].std()
        ax5.errorbar(
            infer_mean.index, infer_mean.values,
            yerr=infer_std.values,
            marker='s', capsize=5, label='RFF-KDE', color='orange'
        )
        ax5.axhline(
            df_kde['Infer_Time_sec'].mean(), color='red', linestyle='--',
            label='Exact KDE'
        )
        ax5.fill_between(
            D_VALUES,
            df_kde['Infer_Time_sec'].mean() - df_kde['Infer_Time_sec'].std(),
            df_kde['Infer_Time_sec'].mean() + df_kde['Infer_Time_sec'].std(),
            color='red', alpha=0.2
        )
        ax5.axhline(
            df_gnb['Infer_Time_sec'].mean(), color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax5.fill_between(
            D_VALUES,
            df_gnb['Infer_Time_sec'].mean() - df_gnb['Infer_Time_sec'].std(),
            df_gnb['Infer_Time_sec'].mean() + df_gnb['Infer_Time_sec'].std(),
            color='green', alpha=0.2
        )
        ax5.set_xlabel('Number of RFF Components (D)')
        ax5.set_ylabel('Inference Time (sec)')
        ax5.set_title('Inference Time vs D')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log')
        ax5.set_yscale('log')
        
        # 2.3: Total Time vs D
        ax6 = fig.add_subplot(gs[1, 2])
        total_mean = df_rff.groupby('D')['Total_Time_sec'].mean()
        total_std  = df_rff.groupby('D')['Total_Time_sec'].std()
        ax6.errorbar(
            total_mean.index, total_mean.values,
            yerr=total_std.values,
            marker='s', capsize=5, label='RFF-KDE', color='purple'
        )
        ax6.axhline(
            df_kde['Total_Time_sec'].mean(), color='red', linestyle='--',
            label='Exact KDE'
        )
        ax6.fill_between(
            D_VALUES,
            df_kde['Total_Time_sec'].mean() - df_kde['Total_Time_sec'].std(),
            df_kde['Total_Time_sec'].mean() + df_kde['Total_Time_sec'].std(),
            color='red', alpha=0.2
        )
        ax6.axhline(
            df_gnb['Total_Time_sec'].mean(), color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax6.fill_between(
            D_VALUES,
            df_gnb['Total_Time_sec'].mean() - df_gnb['Total_Time_sec'].std(),
            df_gnb['Total_Time_sec'].mean() + df_gnb['Total_Time_sec'].std(),
            color='green', alpha=0.2
        )
        ax6.set_xlabel('Number of RFF Components (D)')
        ax6.set_ylabel('Total Time (sec)')
        ax6.set_title('Total Time vs D')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')
        ax6.set_yscale('log')
        
        # ---------------------------------------------------------
        # Row 3: Memory & Efficiency
        # ---------------------------------------------------------
        # 3.1: Training Memory vs D
        ax7 = fig.add_subplot(gs[2, 0])
        train_mem_mean = df_rff.groupby('D')['Train_Memory_MB'].mean()
        train_mem_std  = df_rff.groupby('D')['Train_Memory_MB'].std()
        ax7.errorbar(
            train_mem_mean.index, train_mem_mean.values,
            yerr=train_mem_std.values,
            marker='^', capsize=5, label='RFF-KDE', color='blue'
        )
        ax7.axhline(
            df_kde['Train_Memory_MB'].mean(), color='red', linestyle='--',
            label='Exact KDE'
        )
        ax7.fill_between(
            D_VALUES,
            df_kde['Train_Memory_MB'].mean() - df_kde['Train_Memory_MB'].std(),
            df_kde['Train_Memory_MB'].mean() + df_kde['Train_Memory_MB'].std(),
            color='red', alpha=0.2
        )
        ax7.axhline(
            df_gnb['Train_Memory_MB'].mean(), color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax7.fill_between(
            D_VALUES,
            df_gnb['Train_Memory_MB'].mean() - df_gnb['Train_Memory_MB'].std(),
            df_gnb['Train_Memory_MB'].mean() + df_gnb['Train_Memory_MB'].std(),
            color='green', alpha=0.2
        )
        ax7.set_xlabel('Number of RFF Components (D)')
        ax7.set_ylabel('Training Memory (MB)')
        ax7.set_title('Training Memory Usage vs D')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_xscale('log')
        
        # 3.2: Inference Memory vs D
        ax8 = fig.add_subplot(gs[2, 1])
        infer_mem_mean = df_rff.groupby('D')['Infer_Memory_MB'].mean()
        infer_mem_std  = df_rff.groupby('D')['Infer_Memory_MB'].std()
        ax8.errorbar(
            infer_mem_mean.index, infer_mem_mean.values,
            yerr=infer_mem_std.values,
            marker='^', capsize=5, label='RFF-KDE', color='orange'
        )
        ax8.axhline(
            df_kde['Infer_Memory_MB'].mean(), color='red', linestyle='--',
            label='Exact KDE'
        )
        ax8.fill_between(
            D_VALUES,
            df_kde['Infer_Memory_MB'].mean() - df_kde['Infer_Memory_MB'].std(),
            df_kde['Infer_Memory_MB'].mean() + df_kde['Infer_Memory_MB'].std(),
            color='red', alpha=0.2
        )
        ax8.axhline(
            df_gnb['Infer_Memory_MB'].mean(), color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax8.fill_between(
            D_VALUES,
            df_gnb['Infer_Memory_MB'].mean() - df_gnb['Infer_Memory_MB'].std(),
            df_gnb['Infer_Memory_MB'].mean() + df_gnb['Infer_Memory_MB'].std(),
            color='green', alpha=0.2
        )
        ax8.set_xlabel('Number of RFF Components (D)')
        ax8.set_ylabel('Inference Memory (MB)')
        ax8.set_title('Inference Memory Usage vs D')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_xscale('log')
        
        # 3.3: AUC / Total Time (Efficiency)
        ax9 = fig.add_subplot(gs[2, 2])
        rff_auc_eff   = df_rff.groupby('D')['AUC'].mean() / df_rff.groupby('D')['Total_Time_sec'].mean()
        kde_auc_eff   = df_kde['AUC'].mean() / df_kde['Total_Time_sec'].mean()
        gnb_auc_eff   = df_gnb['AUC'].mean() / df_gnb['Total_Time_sec'].mean()
        
        ax9.plot(
            rff_auc_eff.index, rff_auc_eff.values,
            marker='o', label='RFF-KDE', color='purple'
        )
        ax9.axhline(
            kde_auc_eff, color='red', linestyle='--', label='Exact KDE'
        )
        ax9.axhline(
            gnb_auc_eff, color='green', linestyle=':', label='Gaussian NB'
        )
        ax9.set_xlabel('Number of RFF Components (D)')
        ax9.set_ylabel('Efficiency (AUC / Total Time)')
        ax9.set_title('Computational Efficiency')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_xscale('log')
        
        # ---------------------------------------------------------
        # Row 4: Speedup Analysis
        # ---------------------------------------------------------
        # 4.1: Training Speedup (Exact KDE vs RFF and GNB)
        ax10 = fig.add_subplot(gs[3, 0])
        kde_train_mean = df_kde['Train_Time_sec'].mean()
        rff_train_mean = df_rff.groupby('D')['Train_Time_sec'].mean()
        gnb_train_mean = df_gnb['Train_Time_sec'].mean()
        
        train_speedup_rff = kde_train_mean / rff_train_mean
        train_speedup_gnb = kde_train_mean / gnb_train_mean  # scalar
        
        ax10.plot(
            train_speedup_rff.index, train_speedup_rff.values,
            marker='s', linewidth=2, color='blue', label='RFF-KDE'
        )
        ax10.axhline(
            1.0, color='red', linestyle='--', label='No speedup'
        )
        ax10.axhline(
            train_speedup_gnb, color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax10.set_xlabel('Number of RFF Components (D)')
        ax10.set_ylabel('Speedup (Exact KDE / Method)')
        ax10.set_title('Training Speedup')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        ax10.set_xscale('log')
        
        # 4.2: Inference Speedup
        ax11 = fig.add_subplot(gs[3, 1])
        kde_infer_mean = df_kde['Infer_Time_sec'].mean()
        rff_infer_mean = df_rff.groupby('D')['Infer_Time_sec'].mean()
        gnb_infer_mean = df_gnb['Infer_Time_sec'].mean()
        
        infer_speedup_rff = kde_infer_mean / rff_infer_mean
        infer_speedup_gnb = kde_infer_mean / gnb_infer_mean
        
        ax11.plot(
            infer_speedup_rff.index, infer_speedup_rff.values,
            marker='s', linewidth=2, color='orange', label='RFF-KDE'
        )
        ax11.axhline(
            1.0, color='red', linestyle='--', label='No speedup'
        )
        ax11.axhline(
            infer_speedup_gnb, color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax11.set_xlabel('Number of RFF Components (D)')
        ax11.set_ylabel('Speedup (Exact KDE / Method)')
        ax11.set_title('Inference Speedup')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        ax11.set_xscale('log')
        
        # 4.3: Total Time Speedup
        ax12 = fig.add_subplot(gs[3, 2])
        kde_total_mean = df_kde['Total_Time_sec'].mean()
        rff_total_mean = df_rff.groupby('D')['Total_Time_sec'].mean()
        gnb_total_mean = df_gnb['Total_Time_sec'].mean()
        
        total_speedup_rff = kde_total_mean / rff_total_mean
        total_speedup_gnb = kde_total_mean / gnb_total_mean
        
        ax12.plot(
            total_speedup_rff.index, total_speedup_rff.values,
            marker='s', linewidth=2, color='purple', label='RFF-KDE'
        )
        ax12.axhline(
            1.0, color='red', linestyle='--', label='No speedup'
        )
        ax12.axhline(
            total_speedup_gnb, color='green', linestyle=':',
            label='Gaussian NB'
        )
        ax12.set_xlabel('Number of RFF Components (D)')
        ax12.set_ylabel('Speedup (Exact KDE / Method)')
        ax12.set_title('Total Time Speedup')
        ax12.legend()
        ax12.grid(True, alpha=0.3)
        ax12.set_xscale('log')
        
        # Save figure
        fname = f"img/rff_comparative_analysis_{DIFFICULTY}.png"
        fig.suptitle(f"RFF Comparative Analysis - {DIFFICULTY.capitalize()} Difficulty", fontsize=16)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {fname}")

    plt.show()

def plot_feature_target_correlation(X,y):
    df = pd.DataFrame(X.values, columns=[f"Feat_{i}" for i in range(X.shape[1])])
    df['Class'] = y
    correlation = df.corr()['Class'].drop('Class').sort_values(ascending=False)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Bar plot of correlations
    ax = axes[0]
    colors = ['green' if x > 0 else 'red' for x in correlation.values]
    correlation.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Correlation with Class', fontsize=12)
    ax.set_title('Feature Correlation with Target', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # 2. Heatmap of top correlations
    ax = axes[1]
    top_features = correlation.abs().nlargest(10).index.tolist()
    top_corr_matrix = df[top_features + ['Class']].corr()
    sns.heatmap(top_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                ax=ax, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax.set_title('Top 10 Features Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('img/feature_target_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nFeature-Target Correlation:")
    print(correlation)

def plot_rff_comparative_analysis_magic(df_magic, D_VALUES_MAGIC):
    
    # 2. Extract data by method
    df_kde = df_magic[df_magic['Method'] == 'Exact_KDE'].copy()
    df_gnb = df_magic[df_magic['Method'] == 'Gaussian_NB'].copy()
    df_rff = df_magic[(df_magic['Method'] == 'RFF_KDE') & (df_magic['D'] > 0)].copy()

    # 3. Check data availability
    has_kde = not df_kde.empty
    has_gnb = not df_gnb.empty
    
    if not has_kde:
        print("⚠️  WARNING: No Exact_KDE data found. Baseline will not be plotted.")
    if not has_gnb:
        print("⚠️  WARNING: No Gaussian_NB data found. Baseline will not be plotted.")
    if df_rff.empty:
        print("❌ ERROR: No RFF_KDE data found. Cannot plot.")
        return
    
    # Get unique D values
    D_VALUES_MAGIC = sorted(df_rff['D'].unique())
    
    # Create figure
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # =================================================================
    # Row 1: AUC Analysis
    # =================================================================
    
    # 1.1: AUC vs D
    ax1 = fig.add_subplot(gs[0, 0])
    rff_auc_mean = df_rff.groupby('D')['AUC'].mean()
    rff_auc_std = df_rff.groupby('D')['AUC'].std()
    ax1.errorbar(rff_auc_mean.index, rff_auc_mean.values, yerr=rff_auc_std.values, 
                marker='o', capsize=5, label='RFF-KDE')
    
    if has_kde:
        kde_auc_mean = df_kde['AUC'].mean()
        kde_auc_std = df_kde['AUC'].std()
        ax1.axhline(kde_auc_mean, color='red', linestyle='--', label='Exact KDE')
        ax1.fill_between(D_VALUES_MAGIC, 
                        kde_auc_mean - kde_auc_std,
                        kde_auc_mean + kde_auc_std,
                        color='red', alpha=0.2)
    
    if has_gnb:
        gnb_auc_mean = df_gnb['AUC'].mean()
        gnb_auc_std = df_gnb['AUC'].std()
        ax1.axhline(gnb_auc_mean, color='green', linestyle=':', label='Gaussian NB')
        ax1.fill_between(D_VALUES_MAGIC, 
                        gnb_auc_mean - gnb_auc_std,
                        gnb_auc_mean + gnb_auc_std,
                        color='green', alpha=0.2)
    
    ax1.set_xlabel('Number of RFF Components (D)')
    ax1.set_ylabel('AUC')
    ax1.set_title('AUC vs RFF Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # 1.2: AUC Error
    ax2 = fig.add_subplot(gs[0, 1])
    if has_kde:
        exact_auc_mean = df_kde['AUC'].mean()
        auc_error = df_rff.groupby('D')['AUC'].mean() - exact_auc_mean
        auc_error_std = df_rff.groupby('D')['AUC'].std()
        ax2.errorbar(auc_error.index, auc_error.values, yerr=auc_error_std.values,
                    marker='o', capsize=5, color='green')
        ax2.axhline(0, color='red', linestyle='--', linewidth=2, label='Exact KDE baseline')
        ax2.set_ylabel('AUC Error (RFF - Exact)')
        ax2.set_title('Approximation Error')
    else:
        ax2.text(0.5, 0.5, 'No baseline data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Approximation Error (N/A)')
    
    ax2.set_xlabel('Number of RFF Components (D)')
    ax2.grid(True, alpha=0.3)
    if has_kde:
        ax2.set_xscale('log')
    ax2.legend()

    # 1.3: AUC Boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    df_rff_copy = df_rff.copy()
    df_rff_copy['D_str'] = df_rff_copy['D'].astype(str)
    sns.boxplot(data=df_rff_copy, x='D_str', y='AUC', ax=ax3, color='skyblue')
    
    if has_kde:
        ax3.axhline(df_kde['AUC'].mean(), color='red', linestyle='--', label='Exact KDE')
    if has_gnb:
        ax3.axhline(df_gnb['AUC'].mean(), color='green', linestyle=':', label='Gaussian NB')
    
    ax3.set_xlabel('RFF Components (D)')
    ax3.set_ylabel('AUC')
    ax3.set_title('AUC Distribution')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)

    # =================================================================
    # Row 2: Time Analysis (LOG SCALE - SAFE)
    # =================================================================

    # 2.1: Training Time
    ax4 = fig.add_subplot(gs[1, 0])
    train_mean = df_rff.groupby('D')['Train_Time_sec'].mean()
    train_std = df_rff.groupby('D')['Train_Time_sec'].std()
    ax4.errorbar(train_mean.index, train_mean.values, yerr=train_std.values,
                marker='s', capsize=5, label='RFF-KDE', color='blue')
    
    if has_kde:
        kde_train_mean = df_kde['Train_Time_sec'].mean()
        kde_train_std = df_kde['Train_Time_sec'].std()
        ax4.axhline(kde_train_mean, color='red', linestyle='--', label='Exact KDE')
        lower_train_kde = max(kde_train_mean - kde_train_std, 0)
        ax4.fill_between(D_VALUES_MAGIC, lower_train_kde, 
                        kde_train_mean + kde_train_std, color='red', alpha=0.2)
    
    if has_gnb:
        gnb_train_mean = df_gnb['Train_Time_sec'].mean()
        gnb_train_std = df_gnb['Train_Time_sec'].std()
        ax4.axhline(gnb_train_mean, color='green', linestyle=':', label='Gaussian NB')
        lower_train_gnb = max(gnb_train_mean - gnb_train_std, 0)
        ax4.fill_between(D_VALUES_MAGIC, lower_train_gnb, 
                        gnb_train_mean + gnb_train_std, color='green', alpha=0.2)
    
    ax4.set_xlabel('Number of RFF Components (D)')
    ax4.set_ylabel('Training Time (sec)')
    ax4.set_title('Training Time vs D')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    # 2.2: Inference Time
    ax5 = fig.add_subplot(gs[1, 1])
    infer_mean = df_rff.groupby('D')['Infer_Time_sec'].mean()
    infer_std = df_rff.groupby('D')['Infer_Time_sec'].std()
    ax5.errorbar(infer_mean.index, infer_mean.values, yerr=infer_std.values,
                marker='s', capsize=5, label='RFF-KDE', color='orange')
    
    if has_kde:
        kde_infer_mean = df_kde['Infer_Time_sec'].mean()
        kde_infer_std = df_kde['Infer_Time_sec'].std()
        ax5.axhline(kde_infer_mean, color='red', linestyle='--', label='Exact KDE')
        lower_infer_kde = max(kde_infer_mean - kde_infer_std, 0)
        ax5.fill_between(D_VALUES_MAGIC, lower_infer_kde, 
                        kde_infer_mean + kde_infer_std, color='red', alpha=0.2)
    
    if has_gnb:
        gnb_infer_mean = df_gnb['Infer_Time_sec'].mean()
        gnb_infer_std = df_gnb['Infer_Time_sec'].std()
        ax5.axhline(gnb_infer_mean, color='green', linestyle=':', label='Gaussian NB')
        lower_infer_gnb = max(gnb_infer_mean - gnb_infer_std, 0)
        ax5.fill_between(D_VALUES_MAGIC, lower_infer_gnb, 
                        gnb_infer_mean + gnb_infer_std, color='green', alpha=0.2)
    
    ax5.set_xlabel('Number of RFF Components (D)')
    ax5.set_ylabel('Inference Time (sec)')
    ax5.set_title('Inference Time vs D')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    ax5.set_yscale('log')

    # 2.3: Total Time
    ax6 = fig.add_subplot(gs[1, 2])
    total_mean = df_rff.groupby('D')['Total_Time_sec'].mean()
    total_std = df_rff.groupby('D')['Total_Time_sec'].std()
    ax6.errorbar(total_mean.index, total_mean.values, yerr=total_std.values,
                marker='s', capsize=5, label='RFF-KDE', color='purple')
    
    if has_kde:
        kde_total_mean = df_kde['Total_Time_sec'].mean()
        kde_total_std = df_kde['Total_Time_sec'].std()
        ax6.axhline(kde_total_mean, color='red', linestyle='--', label='Exact KDE')
        lower_total_kde = max(kde_total_mean - kde_total_std, 0)
        ax6.fill_between(D_VALUES_MAGIC, lower_total_kde, 
                        kde_total_mean + kde_total_std, color='red', alpha=0.2)
    
    if has_gnb:
        gnb_total_mean = df_gnb['Total_Time_sec'].mean()
        gnb_total_std = df_gnb['Total_Time_sec'].std()
        ax6.axhline(gnb_total_mean, color='green', linestyle=':', label='Gaussian NB')
        lower_total_gnb = max(gnb_total_mean - gnb_total_std, 0)
        ax6.fill_between(D_VALUES_MAGIC, lower_total_gnb, 
                        gnb_total_mean + gnb_total_std, color='green', alpha=0.2)
    
    ax6.set_xlabel('Number of RFF Components (D)')
    ax6.set_ylabel('Total Time (sec)')
    ax6.set_title('Total Time vs D')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')
    ax6.set_yscale('log')

    # =================================================================
    # Row 3: Memory & Efficiency
    # =================================================================

    # 3.1: Training Memory
    ax7 = fig.add_subplot(gs[2, 0])
    train_mem_mean = df_rff.groupby('D')['Train_Memory_MB'].mean()
    train_mem_std = df_rff.groupby('D')['Train_Memory_MB'].std()
    ax7.errorbar(train_mem_mean.index, train_mem_mean.values, yerr=train_mem_std.values,
                marker='^', capsize=5, label='RFF-KDE', color='blue')
    
    if has_kde:
        kde_mem_mean = df_kde['Train_Memory_MB'].mean()
        kde_mem_std = df_kde['Train_Memory_MB'].std()
        ax7.axhline(kde_mem_mean, color='red', linestyle='--', label='Exact KDE')
        ax7.fill_between(D_VALUES_MAGIC, kde_mem_mean - kde_mem_std,
                        kde_mem_mean + kde_mem_std, color='red', alpha=0.2)
    
    if has_gnb:
        gnb_mem_mean = df_gnb['Train_Memory_MB'].mean()
        gnb_mem_std = df_gnb['Train_Memory_MB'].std()
        ax7.axhline(gnb_mem_mean, color='green', linestyle=':', label='Gaussian NB')
        ax7.fill_between(D_VALUES_MAGIC, gnb_mem_mean - gnb_mem_std,
                        gnb_mem_mean + gnb_mem_std, color='green', alpha=0.2)
    
    ax7.set_xlabel('Number of RFF Components (D)')
    ax7.set_ylabel('Training Memory (MB)')
    ax7.set_title('Training Memory Usage vs D')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xscale('log')

    # 3.2: Inference Memory
    ax8 = fig.add_subplot(gs[2, 1])
    infer_mem_mean = df_rff.groupby('D')['Infer_Memory_MB'].mean()
    infer_mem_std = df_rff.groupby('D')['Infer_Memory_MB'].std()
    ax8.errorbar(infer_mem_mean.index, infer_mem_mean.values, yerr=infer_mem_std.values,
                marker='^', capsize=5, label='RFF-KDE', color='orange')
    
    if has_kde:
        kde_imem_mean = df_kde['Infer_Memory_MB'].mean()
        kde_imem_std = df_kde['Infer_Memory_MB'].std()
        ax8.axhline(kde_imem_mean, color='red', linestyle='--', label='Exact KDE')
        ax8.fill_between(D_VALUES_MAGIC, kde_imem_mean - kde_imem_std,
                        kde_imem_mean + kde_imem_std, color='red', alpha=0.2)
    
    if has_gnb:
        gnb_imem_mean = df_gnb['Infer_Memory_MB'].mean()
        gnb_imem_std = df_gnb['Infer_Memory_MB'].std()
        ax8.axhline(gnb_imem_mean, color='green', linestyle=':', label='Gaussian NB')
        ax8.fill_between(D_VALUES_MAGIC, gnb_imem_mean - gnb_imem_std,
                        gnb_imem_mean + gnb_imem_std, color='green', alpha=0.2)
    
    ax8.set_xlabel('Number of RFF Components (D)')
    ax8.set_ylabel('Inference Memory (MB)')
    ax8.set_title('Inference Memory Usage vs D')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xscale('log')

    # 3.3: Efficiency
    ax9 = fig.add_subplot(gs[2, 2])
    rff_auc_mean_vals = df_rff.groupby('D')['AUC'].mean()
    rff_total_mean_vals = df_rff.groupby('D')['Total_Time_sec'].mean()
    rff_auc_eff = rff_auc_mean_vals / rff_total_mean_vals
    
    ax9.plot(rff_auc_eff.index, rff_auc_eff.values, marker='o', label='RFF-KDE', color='purple')
    
    if has_kde:
        kde_auc_eff = df_kde['AUC'].mean() / df_kde['Total_Time_sec'].mean()
        ax9.axhline(kde_auc_eff, color='red', linestyle='--', label='Exact KDE')
    
    if has_gnb:
        gnb_auc_eff = df_gnb['AUC'].mean() / df_gnb['Total_Time_sec'].mean()
        ax9.axhline(gnb_auc_eff, color='green', linestyle=':', label='Gaussian NB')
    
    ax9.set_xlabel('Number of RFF Components (D)')
    ax9.set_ylabel('Efficiency (AUC / Total Time)')
    ax9.set_title('Computational Efficiency')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_xscale('log')

    # =================================================================
    # Row 4: Speedup Analysis
    # =================================================================

    # 4.1: Training Speedup
    ax10 = fig.add_subplot(gs[3, 0])
    if has_kde:
        kde_train_mean = df_kde['Train_Time_sec'].mean()
        rff_train_mean = df_rff.groupby('D')['Train_Time_sec'].mean()
        train_speedup_rff = kde_train_mean / rff_train_mean
        ax10.plot(train_speedup_rff.index, train_speedup_rff.values, 
                 marker='s', linewidth=2, color='blue', label='RFF-KDE')
        ax10.axhline(1.0, color='red', linestyle='--', label='No speedup')
        
        if has_gnb:
            gnb_train_mean = df_gnb['Train_Time_sec'].mean()
            train_speedup_gnb = kde_train_mean / gnb_train_mean
            ax10.axhline(train_speedup_gnb, color='green', linestyle=':', label='Gaussian NB')
    else:
        ax10.text(0.5, 0.5, 'No baseline data available', 
                 ha='center', va='center', transform=ax10.transAxes, fontsize=14)
    
    ax10.set_xlabel('Number of RFF Components (D)')
    ax10.set_ylabel('Speedup (Exact KDE / Method)')
    ax10.set_title('Training Speedup')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    if has_kde:
        ax10.set_xscale('log')

    # 4.2: Inference Speedup
    ax11 = fig.add_subplot(gs[3, 1])
    if has_kde:
        kde_infer_mean = df_kde['Infer_Time_sec'].mean()
        rff_infer_mean = df_rff.groupby('D')['Infer_Time_sec'].mean()
        infer_speedup_rff = kde_infer_mean / rff_infer_mean
        ax11.plot(infer_speedup_rff.index, infer_speedup_rff.values, 
                 marker='s', linewidth=2, color='orange', label='RFF-KDE')
        ax11.axhline(1.0, color='red', linestyle='--', label='No speedup')
        
        if has_gnb:
            gnb_infer_mean = df_gnb['Infer_Time_sec'].mean()
            infer_speedup_gnb = kde_infer_mean / gnb_infer_mean
            ax11.axhline(infer_speedup_gnb, color='green', linestyle=':', label='Gaussian NB')
    else:
        ax11.text(0.5, 0.5, 'No baseline data available', 
                 ha='center', va='center', transform=ax11.transAxes, fontsize=14)
    
    ax11.set_xlabel('Number of RFF Components (D)')
    ax11.set_ylabel('Speedup (Exact KDE / Method)')
    ax11.set_title('Inference Speedup')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    if has_kde:
        ax11.set_xscale('log')

    # 4.3: Total Time Speedup
    ax12 = fig.add_subplot(gs[3, 2])
    if has_kde:
        kde_total_mean = df_kde['Total_Time_sec'].mean()
        rff_total_mean = df_rff.groupby('D')['Total_Time_sec'].mean()
        total_speedup_rff = kde_total_mean / rff_total_mean
        ax12.plot(total_speedup_rff.index, total_speedup_rff.values, 
                 marker='s', linewidth=2, color='purple', label='RFF-KDE')
        ax12.axhline(1.0, color='red', linestyle='--', label='No speedup')
        
        if has_gnb:
            gnb_total_mean = df_gnb['Total_Time_sec'].mean()
            total_speedup_gnb = kde_total_mean / gnb_total_mean
            ax12.axhline(total_speedup_gnb, color='green', linestyle=':', label='Gaussian NB')
    else:
        ax12.text(0.5, 0.5, 'No baseline data available', 
                 ha='center', va='center', transform=ax12.transAxes, fontsize=14)
    
    ax12.set_xlabel('Number of RFF Components (D)')
    ax12.set_ylabel('Speedup (Exact KDE / Method)')
    ax12.set_title('Total Time Speedup')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    if has_kde:
        ax12.set_xscale('log')

    # Save figure
    fname = "img/rff_comparative_analysis_magic_gamma.png"
    fig.suptitle("RFF Comparative Analysis - Magic Gamma Dataset", fontsize=16)
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {fname}")

    plt.show()


def plot_rff_comparative_analysis_scaling(results_scaling):
    df_scaling = pd.DataFrame(results_scaling)

    # Create figure with 1 row x 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------------------------------------------------
    # 1. Inference Time vs Dataset Size
    # ---------------------------------------------------------
    ax = axes[0]
    for method in ['Exact_KDE', 'RFF_KDE']:
        df_method = df_scaling[df_scaling['Method'] == method]
        infer_mean = df_method.groupby('Size')['Infer_Time_sec'].mean()
        infer_std = df_method.groupby('Size')['Infer_Time_sec'].std()
        
        if method == 'Exact_KDE':
            label = 'Exact KDE'
            color = 'red'
            marker = 'o'
        else:
            label = 'RFF-KDE (D=log(n))'
            color = 'blue'
            marker = 's'
        
        ax.errorbar(infer_mean.index, infer_mean.values, yerr=infer_std.values,
                    marker=marker, capsize=5, label=label, linewidth=2, color=color)

    ax.set_xlabel('Dataset Size (n)', fontsize=12)
    ax.set_ylabel('Inference Time (sec)', fontsize=12)
    ax.set_title('Inference Time vs Dataset Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # 2. Training Time vs Dataset Size
    # ---------------------------------------------------------
    ax = axes[1]
    for method in ['Exact_KDE', 'RFF_KDE']:
        df_method = df_scaling[df_scaling['Method'] == method]
        train_mean = df_method.groupby('Size')['Train_Time_sec'].mean()
        train_std = df_method.groupby('Size')['Train_Time_sec'].std()
        
        if method == 'Exact_KDE':
            label = 'Exact KDE'
            color = 'red'
            marker = 'o'
        else:
            label = 'RFF-KDE (D=log(n))'
            color = 'blue'
            marker = 's'
        
        ax.errorbar(train_mean.index, train_mean.values, yerr=train_std.values,
                    marker=marker, capsize=5, label=label, linewidth=2, color=color)

    ax.set_xlabel('Dataset Size (n)', fontsize=12)
    ax.set_ylabel('Training Time (sec)', fontsize=12)
    ax.set_title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Scaling Analysis: Exact KDE vs RFF-KDE (D=log(n))', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    fname = 'img/rff_comparative_analysis_scaling.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {fname}")

    plt.show()

def scaling_summary_table(results_scaling, sizes):
    df_scaling = pd.DataFrame(results_scaling)
    summary_table = []
    for size in sizes:
        df_size = df_scaling[df_scaling['Size'] == size]
        df_kde_size = df_size[df_size['Method'] == 'Exact_KDE']
        df_rff_size = df_size[df_size['Method'] == 'RFF_KDE']
        
        D_value = df_rff_size['D'].iloc[0]
        
        summary_table.append({
            'Size': size,
            'D': D_value,
            'KDE_Train_Time': df_kde_size['Train_Time_sec'].mean(),
            'RFF_Train_Time': df_rff_size['Train_Time_sec'].mean(),
            'KDE_Infer_Time': df_kde_size['Infer_Time_sec'].mean(),
            'RFF_Infer_Time': df_rff_size['Infer_Time_sec'].mean(),
            'Train_Speedup': df_kde_size['Train_Time_sec'].mean() / df_rff_size['Train_Time_sec'].mean(),
            'Infer_Speedup': df_kde_size['Infer_Time_sec'].mean() / df_rff_size['Infer_Time_sec'].mean(),
            'KDE_AUC': df_kde_size['AUC'].mean(),
            'RFF_AUC': df_rff_size['AUC'].mean(),
            'AUC_Diff': df_rff_size['AUC'].mean() - df_kde_size['AUC'].mean()
        })

    summary_df = pd.DataFrame(summary_table)
    print(summary_df.round(4).to_string(index=False))

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print(f"As dataset size increases from {min(sizes)} to {max(sizes)}:")
    print(f"- RFF-KDE uses D=log(n) features, ranging from {summary_df['D'].min()} to {summary_df['D'].max()}")
    print(f"- Training speedup ranges from {summary_df['Train_Speedup'].min():.2f}x to {summary_df['Train_Speedup'].max():.2f}x")
    print(f"- Inference speedup ranges from {summary_df['Infer_Speedup'].min():.2f}x to {summary_df['Infer_Speedup'].max():.2f}x")
    print(f"- AUC difference (RFF - KDE) ranges from {summary_df['AUC_Diff'].min():.4f} to {summary_df['AUC_Diff'].max():.4f}")
    print("="*80)
