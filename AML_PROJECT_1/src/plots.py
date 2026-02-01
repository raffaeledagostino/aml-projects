import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import numerical_cols, special_cols
from scipy import stats

###################### PLOT CONTINUOUS VARIABLES WITH Q-Q PLOTS ######################
def plot_continuous_distributions_with_qq(dataset):
    """
    Plot barplots, KDE plots, and Q-Q plots for all numerical variables comparing target classes.
    Each row contains: barplot, KDE plot, Q-Q plot for class 0, Q-Q plot for class 1.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset to visualize (must contain 'num' column as target)
    """
    continuous_cols = list(numerical_cols.keys()) + list(special_cols.keys())
    
    # Set color palette for target classes (0=no disease, 1=disease)
    sns.set_palette(['#ff826e', 'red'])
    
    # Create the subplots (one row per feature, 4 columns: barplot, KDE, Q-Q for class 0, Q-Q for class 1)
    fig, ax = plt.subplots(len(continuous_cols), 4, figsize=(25, 5*len(continuous_cols)), 
                          gridspec_kw={'width_ratios': [1, 2, 1.5, 1.5]})
    
    # Handle case where there's only one numerical feature
    if len(continuous_cols) == 1:
        ax = ax.reshape(1, -1)
    
    # Loop through each numerical feature
    for i, col in enumerate(continuous_cols):
        # 1. Barplot showing the mean value of the feature for each target category
        graph = sns.barplot(data=dataset, x="num", y=col, ax=ax[i, 0])
        ax[i, 0].set_xlabel('Heart Disease', fontsize=11)
        ax[i, 0].set_ylabel(col, fontsize=11)
        
        # Add mean values as labels on the barplot
        for cont in graph.containers:
            graph.bar_label(cont, fmt='         %.3g')
        
        # 2. KDE plot showing the distribution of the feature for each target category
        sns.kdeplot(data=dataset[dataset["num"]==0], x=col, fill=True, linewidth=2, 
                   ax=ax[i, 1], label='0')
        sns.kdeplot(data=dataset[dataset["num"]==1], x=col, fill=True, linewidth=2, 
                   ax=ax[i, 1], label='1')
        ax[i, 1].set_yticks([])  # Remove y-axis ticks (density values not needed)
        ax[i, 1].set_xlabel(col, fontsize=11)
        ax[i, 1].legend(title='Heart Disease', loc='upper right')
        
        # 3. Q-Q plot for class 0 (no disease)
        stats.probplot(dataset[dataset["num"]==0][col].dropna(), dist="norm", plot=ax[i, 2])
        ax[i, 2].set_title(f'{col} - Class 0 (No Disease)', fontsize=10, fontweight='bold')
        ax[i, 2].get_lines()[0].set_color('#ff826e')
        ax[i, 2].get_lines()[0].set_markersize(4)
        
        # 4. Q-Q plot for class 1 (disease)
        stats.probplot(dataset[dataset["num"]==1][col].dropna(), dist="norm", plot=ax[i, 3])
        ax[i, 3].set_title(f'{col} - Class 1 (Disease)', fontsize=10, fontweight='bold')
        ax[i, 3].get_lines()[0].set_color('red')
        ax[i, 3].get_lines()[0].set_markersize(4)

        
    
    # Set the title for the entire figure
    #plt.suptitle('Numerical Features: Distribution and Normality Analysis', fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig('img/continuous_distributions_with_qq.png', dpi=300, bbox_inches='tight')
    plt.show()

######################## CORRELATION PLOT HEATMAP #####################
def plot_correlation_heatmap(dataset):
    """
    Plots a heatmap of the correlation matrix for numerical features in the dataset.

    Parameters:
    -----------
    dataset : pd.DataFrame
        Dataset containing numerical features
    """

    continuous_cols = list(numerical_cols.keys()) + list(special_cols.keys())

    # Compute correlation matrix
    corr = dataset[continuous_cols].corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Set title
    # plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
    
    # Save figure to file
    plt.savefig("img/correlation_heatmap.png", dpi=100, bbox_inches='tight')
    plt.show()

###################### BOXPLOT OF THE AUC VALUES ######################
def plot_auc_boxplots(results):
    """
    Plots boxplots of AUC scores for different models.

    Parameters:
    -----------
    results : dict
        Dictionary containing model statistics with 'auc_scores' list for each model
    """
    # Prepare data for plotting
    model_names = list(results.keys())
    auc_data = [results[model]['auc_scores'] for model in model_names]

    # Create boxplot figure
    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(auc_data, labels=model_names, patch_artist=True)
    
    # Customize colors for each model
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Set axis labels and title
    plt.ylabel('AUC Scores', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    # plt.title('AUC Score Distribution by Model', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=15, ha='right')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure to file
    plt.savefig("img/auc_boxplots.png", dpi=100)
    plt.show()

############ DISTRIBUTION OF THE HYPERPARAMETERS VALUES ###############

def plot_hyperparameter_distributions(results):
    """
    Plots the distributions of hyperparameter values for different models in separate subplots.
    Uses histograms for continuous hyperparameters and barplots for discrete ones.

    Parameters:
    -----------
    results : dict
        Dictionary containing model statistics with 'auc_scores' and optionally 'best_param' lists
    """
    # Define hyperparameter names and colors for each model
    hyperparameter_info = {
        'Logistic Regression': {'name': 'C (Regularization)', 'color': 'lightblue', 'type': 'continuous'},
        'Decision Trees': {'name': 'Max Depth', 'color': 'lightgreen', 'type': 'discrete'},
        'SVM': {'name': 'C (Regularization)', 'color': 'lightcoral', 'type': 'continuous'}
    }
    
    # Select only models that have hyperparameter data to plot
    models_to_plot = [m for m in results.keys() if 'best_param' in results[m] and results[m]['best_param']]
    
    # Get number of models to plot
    n_models = len(models_to_plot)
    if n_models == 0:
        print("No hyperparameter data to plot.")
        return
    
    # Create subplots (one for each model with hyperparameters)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    if n_models == 1:
        axes = [axes]  # Make iterable for single plot
    
    # Plot histogram or barplot for each model
    for ax, model_name in zip(axes, models_to_plot):
        # Get hyperparameter values and filter out None values
        best_params = results[model_name]['best_param']
        best_params_filtered = [p for p in best_params if p is not None]
        
        # Get hyperparameter name, color, and type from info dictionary
        hyperparam_name = hyperparameter_info[model_name]['name']
        color = hyperparameter_info[model_name]['color']
        param_type = hyperparameter_info[model_name]['type']
        
        # Only plot if there are valid hyperparameter values
        if best_params_filtered:
            if param_type == 'discrete':
                # For discrete parameters (like max_depth), use barplot
                unique_vals, counts = np.unique(best_params_filtered, return_counts=True)
                ax.bar(unique_vals, counts, alpha=0.7, color=color, edgecolor='black', width=0.6)
                
                # Set x-axis to show integer values only
                ax.set_xticks(unique_vals)
                
            else:
                # For continuous parameters, use histogram
                ax.hist(best_params_filtered, bins=20, alpha=0.7, color=color, edgecolor='black')
        
        # Set labels and title
        ax.set_xlabel(hyperparam_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
    
    # Add overall title
    # plt.suptitle('Distribution of Best Hyperparameter Values', fontsize=14, fontweight='bold', y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure to file
    plt.savefig("img/hyperparameter_distributions.png", dpi=100, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    pass