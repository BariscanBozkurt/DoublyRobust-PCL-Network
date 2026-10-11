import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import Math, display, HTML

def format_results_for_jupyter_and_latex(df: pd.DataFrame,
                                         dataset_col: str = 'Dataset',
                                         size_col: str = 'Data_Size',
                                         algo_col: str = 'Algorithm',
                                         metric_col: str = 'Causal_MSE',
                                         precision: int = 2,
                                         print_latex_code = False):

    # 1. Ensure dataset column exists
    if dataset_col not in df.columns:
        df[dataset_col] = dataset_col

    # 2. Calculate Mean and Standard Deviation
    agg_df = df.groupby([dataset_col, size_col, algo_col])[metric_col].agg(['mean', 'std']).reset_index()

    # 3. Pivot the data
    mean_pivot = agg_df.pivot(index=[dataset_col, size_col], columns=algo_col, values='mean')
    std_pivot = agg_df.pivot(index=[dataset_col, size_col], columns=algo_col, values='std')
    algorithms = mean_pivot.columns.tolist()

    # ==========================================
    # PART A: Format for Jupyter Visual Display
    # ==========================================
    jupyter_df = pd.DataFrame(index=mean_pivot.index, columns=algorithms)
    latex_strings_dict = {} # Store formatted strings for LaTeX generation later

    for idx in mean_pivot.index:
        row_means = mean_pivot.loc[idx]
        min_mean = row_means.min()
        latex_strings_dict[idx] = {}

        for algo in algorithms:
            m = mean_pivot.loc[idx, algo]
            s = std_pivot.loc[idx, algo]

            if pd.isna(m):
                jupyter_df.loc[idx, algo] = "-"
                latex_strings_dict[idx][algo] = "-"
            else:
                s_val = 0.0 if pd.isna(s) else s

                # HTML version for Jupyter
                html_str = f"{m:.{precision}f} &plusmn; {s_val:.{precision}f}"
                # LaTeX version
                tex_str = f"{m:.{precision}f} $\\pm$ {s_val:.{precision}f}"

                # Bold the best performing algorithm
                if m == min_mean:
                    html_str = f"<b>{html_str}</b>"
                    tex_str = f"\\textbf{{{tex_str}}}"

                jupyter_df.loc[idx, algo] = html_str
                latex_strings_dict[idx][algo] = tex_str

    # Display in Jupyter
    print("--- Jupyter Visual Preview ---")
    # We use HTML to render the bolding and ± signs properly in the dataframe
    display(HTML(jupyter_df.to_html(escape=False)))
    print("\n")

    # ==========================================
    # PART B: Generate Raw LaTeX Code
    # ==========================================
    latex_str = "% Remember to include \\usepackage{multirow} in your LaTeX preamble!\n"
    latex_str += "\\begin{table*}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{cMSE of all methods on synthetic and real-world data.}\n"
    latex_str += "\\begin{tabular}{c|c|" + "c" * len(algorithms) + "}\n"
    latex_str += "\\hline\\hline\n"
    latex_str += "Dataset & Size & " + " & ".join([algo.replace('_', '\\_') for algo in algorithms]) + " \\\\\n"
    latex_str += "\\hline\n"

    current_dataset = None
    for idx in mean_pivot.index:
        dataset, size = idx

        # Handle \multirow
        if dataset != current_dataset:
            num_rows = len(mean_pivot.loc[dataset])
            dataset_str = f"\\multirow{{{num_rows}}}{{*}}{{{dataset}}}"
            current_dataset = dataset
        else:
            dataset_str = ""

        row_vals = " & ".join([latex_strings_dict[idx][algo] for algo in algorithms])
        latex_str += f"{dataset_str} & {size} & {row_vals} \\\\\n"

    latex_str += "\\hline\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table*}\n"
    if print_latex_code:
        print("--- Copy this LaTeX code to your paper ---")
        print(latex_str)

def display_matrix(array: np.ndarray) -> None:
    """
    Display the given numpy array with Latex format in Jupyter Notebook.

    Parameters:
        array (np.ndarray): Array to be displayed.
    """
    data = ""
    for line in array:
        if len(line) == 1:
            data += " %.3f &" % line + r" \\\n"
            continue
        for element in line:
            data += " %.3f &" % element
        data += r" \\" + "\n"
    display(Math("\\begin{bmatrix} \n%s\\end{bmatrix}" % data))

def plot_regression_dashboard(
    target: Union[pd.DataFrame, np.ndarray, list],
    pred: Union[pd.DataFrame, np.ndarray, list],
    log_scale: bool = False,
    figsize: Tuple[int, int] = (16, 9),
    title: str = "Regression Diagnostics",
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    legend_fontsize: int = 12
) -> None:
    """
    Creates a professional diagnostic dashboard for regression analysis.

    Layout:
      - Left Panel: Scatter Plot (Square Aspect Ratio). Visualizes correlation and bias.
      - Right Top Panel: Sorted Curve. Visualizes distributional fit and tail behavior.
      - Right Bottom Panel: Residuals. Visualizes heteroscedasticity, aligned with the sorted curve.

    Parameters:
        target (array-like): Ground truth target values.
        pred (array-like): Predicted values from the model.
        log_scale (bool): If True, applies log-scale to Scatter and Sorted plots.
                          Useful for targets spanning orders of magnitude (e.g., density ratios).
        figsize (tuple): Figure dimensions (width, height). Default (16, 9).
        title (str): Main title of the dashboard.

    Examples:
        >>> # 1. Generate Synthetic Data
        >>> import numpy as np
        >>> N = 1000
        >>> X = np.random.uniform(0, 10, N)
        >>> # Ground Truth: y = 2x + 1
        >>> target = 2 * X + 1
        >>> # Prediction: Truth + Gaussian Noise + Slight Non-linearity bias
        >>> noise = np.random.normal(0, 1.5, N)
        >>> pred = target + noise + (0.02 * target**2)

        >>> # 2. Run Dashboard
        >>> plot_regression_dashboard(target, pred, title="Synthetic Linear Model")
    """

    # 1. Sanitization & Metrics
    # Flatten inputs to 1D arrays
    target = np.array(target).flatten()
    pred = np.array(pred).flatten()

    # Check for NaNs
    mask = ~np.isnan(target) & ~np.isnan(pred)
    target = target[mask]
    pred = pred[mask]

    mse = mean_squared_error(target, pred)
    mae = mean_absolute_error(target, pred)
    r2 = r2_score(target, pred)

    # 2. Sort Data (for Right-hand plots)
    # Sorting by Target allows us to see how the model behaves at different magnitudes
    sort_idx = np.argsort(target)
    sorted_gt = target[sort_idx]
    sorted_pred = pred[sort_idx]
    residuals = sorted_pred - sorted_gt

    # 3. Setup Figure Layout
    # Width Ratios: [1.2, 1] gives the scatter plot (left) more room to stay square
    # Height Ratios: [2, 1] gives the Sorted Curve more height than the Residuals
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[2, 1])

    # Global Title
    fig.suptitle(f"{title} | RMSE: {np.sqrt(mse):.3f} | MAE: {mae:.3f} | R²: {r2:.3f}",
                 fontsize=title_fontsize, fontweight='bold', y=1.03)

    # =========================================================================
    # PLOT 1: SCATTER (Left Column, Spans Both Rows)
    # =========================================================================
    ax_scatter = fig.add_subplot(gs[:, 0])

    # Determine Limits with a small buffer
    low = min(target.min(), pred.min())
    high = max(target.max(), pred.max())
    buff = (high - low) * 0.05

    # Scatter points
    ax_scatter.scatter(target, pred, alpha=0.5, s=20, color='#1f77b4', label='Data')

    # Perfect Fit Line (y=x)
    ax_scatter.plot([low, high], [low, high], 'k--', lw=2, label='Perfect Fit')

    ax_scatter.set_title("Correlation (Scatter)", fontsize=label_fontsize, fontweight='bold')
    ax_scatter.set_xlabel("Target", fontsize=label_fontsize)
    ax_scatter.set_ylabel("Predicted", fontsize=label_fontsize)
    ax_scatter.legend(loc='upper left', fontsize=legend_fontsize)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.tick_params(labelsize=tick_fontsize)

    # Force Square Aspect Ratio
    # adjustable='box' ensures limits are respected while changing the physical plot shape
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.set_xlim(low - buff, high + buff)
    ax_scatter.set_ylim(low - buff, high + buff)

    if log_scale:
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')

    # =========================================================================
    # PLOT 2: SORTED CURVE (Top Right)
    # =========================================================================
    ax_sorted = fig.add_subplot(gs[0, 1])

    # Ground Truth line
    ax_sorted.plot(sorted_gt, color='black', lw=2.5, label='Target (Sorted)', zorder=1)

    # Prediction scatter (dots)
    ax_sorted.scatter(np.arange(len(sorted_pred)), sorted_pred,
                      color='#ff7f0e', s=10, alpha=0.6, label='Prediction', zorder=2)

    ax_sorted.set_title("Distribution Fit (Quantiles)", fontsize=label_fontsize, fontweight='bold')
    ax_sorted.set_ylabel("Value", fontsize=label_fontsize)
    ax_sorted.legend(fontsize=legend_fontsize)
    ax_sorted.grid(True, alpha=0.3)
    ax_sorted.tick_params(labelsize=tick_fontsize)

    # Hide x-labels (shared with residuals)
    ax_sorted.set_xticklabels([])

    if log_scale:
        ax_sorted.set_yscale('log')

    # =========================================================================
    # PLOT 3: RESIDUALS (Bottom Right)
    # =========================================================================
    # Share X-axis with Sorted Curve so zooming matches
    ax_resid = fig.add_subplot(gs[1, 1], sharex=ax_sorted)

    # Zero line
    ax_resid.axhline(0, color='black', linestyle='--', lw=2)

    # Residual dots
    ax_resid.scatter(np.arange(len(residuals)), residuals, color='#d62728', s=10, alpha=0.5)

    ax_resid.set_title("Residuals (Aligned)", fontsize=label_fontsize, fontweight='bold')
    ax_resid.set_ylabel("Error", fontsize=label_fontsize)
    ax_resid.set_xlabel("Sample Index (Sorted)", fontsize=label_fontsize)
    ax_resid.grid(True, alpha=0.3)
    ax_resid.tick_params(labelsize=tick_fontsize)

    plt.show()

def plot_observational_vs_causal_effect(
    A: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    do_A: Union[np.ndarray, torch.Tensor],
    EY_do_A: Union[np.ndarray, torch.Tensor],
    title: str = "Observational Data vs. True Causal Effect",
    figsize: tuple = (10, 6)
):
    """
    Visualizes the biased observational data against the true structural causal function.

    Parameters
    ----------
    A : array-like
        Observational treatment variable.
    Y : array-like
        Observational outcome variable.
    do_A : array-like
        Grid of interventional treatment values.
    EY_do_A : array-like
        True structural causal effect E[Y | do(a)].
    title : str, default="Observational Data vs. True Causal Effect"
        Title of the plot.
    figsize : tuple, default=(10, 6)
        Size of the matplotlib figure.
    """

    # Helper to safely convert PyTorch tensors to flat Numpy arrays
    def to_numpy(x):
        if x is None: return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().flatten()
        return np.array(x).flatten()

    A, Y = to_numpy(A), to_numpy(Y)
    do_A, EY_do_A = to_numpy(do_A), to_numpy(EY_do_A)

    plt.figure(figsize=figsize)

    # 1. Plot the biased observational scatter
    plt.scatter(A, Y, alpha=0.3, s=20, color="slategray", edgecolors="none", label="Observational Data (A, Y)")

    # 2. Plot the true causal curve
    # Sorting is critical to ensure the line plots smoothly from left to right
    sort_idx = np.argsort(do_A)
    plt.plot(do_A[sort_idx], EY_do_A[sort_idx], linewidth=4, color="crimson",
             linestyle="dashed", label="True Causal Effect $E[Y|do(a)]$")

    # 3. Formatting
    plt.title(title, fontsize=16, weight='bold', pad=15)
    plt.xlabel("Treatment (A)", fontsize=14)
    plt.ylabel("Outcome (Y)", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.show()

def plot_heterogeneous_binary_effect(
    V: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    A: Union[np.ndarray, torch.Tensor],
    grid_V: Union[np.ndarray, torch.Tensor],
    f_1_v: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """
    Visualizes binary heterogeneous effects by plotting observational
    outcomes (A=0 vs A=1) against the true causal function f(1, v).
    """

    def to_numpy(x):
        if x is None: return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().flatten()
        return np.array(x).flatten()

    V, Y, A = to_numpy(V), to_numpy(Y), to_numpy(A)
    grid_V, f_1_v = to_numpy(grid_V), to_numpy(f_1_v)

    plt.figure(figsize=figsize)

    # 1. Plot observational data for both groups
    # Masking to separate Treated (A=1) and Control (A=0)
    treated_mask = (A == 1)
    control_mask = (A == 0)

    plt.scatter(V[control_mask], Y[control_mask], color="slategray", alpha=0.3,
                s=20, label="Observed Control ($A=0$)")
    plt.scatter(V[treated_mask], Y[treated_mask], color="royalblue", alpha=0.5,
                s=25, label="Observed Treated ($A=1$)")

    # 2. Plot the CATE curve for f(1, v)
    sort_idx = np.argsort(grid_V)
    plt.plot(grid_V[sort_idx], f_1_v[sort_idx], linewidth=4, color="crimson",
             linestyle="-", label=r"True Causal Function $f_{\text{CATE}}(1, v)$")

    # 3. Formatting
    if title is None:
        title = r"Heterogeneous Treatment Effect: $f_{\text{CATE}}(1, v)$ vs. Observation"

    plt.title(title, fontsize=16, weight='bold', pad=15)
    plt.xlabel("Heterogeneity Variable ($V$)", fontsize=14)
    plt.ylabel("Outcome ($Y$)", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.show()

def plot_causal_effect_estimation(
    do_A: Union[np.ndarray, torch.Tensor],
    EY_do_A: Union[np.ndarray, torch.Tensor],
    f_struct_pred: Union[np.ndarray, torch.Tensor],
    title: str = "Estimated vs. True Causal Effect",
    figsize: tuple = (10, 6),
    print_metrics: bool = True
):
    """
    Plots the predicted structural causal function against the ground truth
    and computes performance metrics (MSE, MAE).

    Parameters
    ----------
    do_A : array-like
        Grid of interventional treatment values.
    EY_do_A : array-like
        True structural causal effect E[Y | do(a)].
    f_struct_pred : array-like
        Model's predicted causal effect.
    title : str, default="Estimated vs. True Causal Effect"
        Title of the plot.
    figsize : tuple, default=(10, 6)
        Size of the matplotlib figure.
    """

    # Helper to safely convert inputs to flat Numpy arrays
    def to_numpy(x):
        if x is None: return None
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy().flatten()
        return np.array(x).flatten()

    do_A = to_numpy(do_A)
    EY_do_A = to_numpy(EY_do_A)
    f_struct_pred = to_numpy(f_struct_pred)

    # Calculate Metrics
    mse = np.mean((f_struct_pred - EY_do_A) ** 2)
    mae = np.mean(np.abs(f_struct_pred - EY_do_A))

    if print_metrics:
        print(f"Structured function test set MSE: {mse:.6f}")
        print(f"Structured function test set MAE: {mae:.6f}")

    # Sort values based on do_A to ensure smooth line plotting
    sort_idx = np.argsort(do_A)
    do_A_sorted = do_A[sort_idx]
    EY_do_A_sorted = EY_do_A[sort_idx]
    f_struct_pred_sorted = f_struct_pred[sort_idx]

    # Create Plot
    plt.figure(figsize=figsize)

    # Plot True Causal Effect
    plt.plot(do_A_sorted, EY_do_A_sorted, linewidth=4, color="crimson",
             linestyle="dashed", label="True $E[Y|do(a)]$")

    # Plot Estimated Causal Effect
    plt.plot(do_A_sorted, f_struct_pred_sorted, linewidth=4, color="royalblue",
             alpha=0.8, label=f"Estimated $E[Y|do(a)]$\n(MSE: {mse:.4f})")

    # Formatting
    plt.title(title, fontsize=16, weight='bold', pad=15)
    plt.xlabel("Treatment (a)", fontsize=14)
    plt.ylabel("Causal Effect", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.show()


def perc(data):
    median = np.zeros(data.shape[1])
    perc_25 = np.zeros(data.shape[1])
    perc_75 = np.zeros(data.shape[1])
    std_data = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.mean(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
        std_data[i] = np.std(data[:, i])

    return median, perc_25, perc_75, std_data

def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)

def ApplyFont(ax, xlabel_text_size = 25.0, ylabel_text_size = 25.0, title_text_size = 19.0, ticks_text_size = 20):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()
    text_size = ticks_text_size

    for t in ticks:
        t.set_fontname('DejaVu Sans')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('DejaVu Sans')
    txt_obj.set_fontsize(xlabel_text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname('DejaVu Sans')
    txt_obj.set_fontsize(ylabel_text_size)

    # txt = ax.get_xticks
    # txt_obj = txt
    # # txt_obj.set_fontname('DejaVu Sans')
    # txt_obj.set_fontsize(x_ticks_text_size)

    # txt = ax.get_yticks
    # txt_obj.set_fontname('DejaVu Sans')
    # txt_obj.set_fontsize(yticks_text_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('DejaVu Sans')
    txt_obj.set_fontsize(title_text_size)

# --- Global Configuration ---
LEGEND_ORDER = [
    'DRPCLNET (V1)',
    'DRPCLNET (V2)',
    'OutcomePCLNet',
    'TreatmentPCLNet',
    'PKDR',
    'PKDR-h',
    'PKDR-q',
    'NMMR-U',
    'NMMR-V',
    'DRKPV',
    'DRKPV (Nystrom)',
    'KPV (Nystrom)',
    'KAP (Nystrom)',
    'KPV',
    'KPV (with X)',
    'OutcomePCLNet (orig)',
    'DRPCLNET (V1) (orig)',
    'DRPCLNET (V2) (orig)',
    'KAP',
    # BridgeRoutePreference 20x3 study (Scenario B / Scenario O): these are the RAW Algorithm names
    # written by Simulations/BridgeRoutePreference/scenario_*.py. Without them _preprocess_pcl_df turns
    # every one of these eight series into NaN and they vanish from the figure with NO error.
    # No other Results CSV carries these raw names into the plotter (the other analysis notebooks map
    # their rows onto the 'OutcomePCLNet'/'DRPCLNET (V1)' names first via their own PAPER dict), so
    # adding them cannot change any existing figure.
    'Naive',
    'OutcomeNet',
    'TreatmentNet',
    'DRPCLNET',
    'OracleOM',
    'OracleTM',
    'DR_OracleH',
    'DR_OraclePhi'
]

COLOR_PALETTE = {
    'DRPCLNET (V1)': '#d62728',      # Red
    'DRPCLNET (V2)': '#e377c2',      # Pink
    'OutcomePCLNet': '#1f77b4',      # Blue
    'TreatmentPCLNet': '#ff7f0e',    # Orange
    'PKDR': '#7f7f7f',               # Gray -- the PKDR doubly robust estimator
    'PKDR-h': '#17becf',             # Cyan -- PKDR outcome-bridge-only arm
    'PKDR-q': '#bcbd22',             # Olive -- PKDR treatment-bridge-only arm
    'NMMR-U': '#393b79',             # Indigo -- NMMR U-statistic loss (their headline arm)
    'NMMR-V': '#a55194',             # Plum -- NMMR V-statistic loss
    'DRKPV': '#2ca02c',              # Green
    'DRKPV (Nystrom)': '#98df8a',      # Light green
    'KPV (Nystrom)': '#c49c94',        # Light brown -- Nystrom outcome bridge (dSprite kernel baseline)
    'KAP (Nystrom)': '#dbdb8d',        # Light olive -- Nystrom treatment bridge (dSprite kernel baseline)
    'KPV': '#9467bd',                # Purple
    'KAP': '#8c564b',                # Brown -- the alternative-proxy (treatment) kernel bridge
    'KPV (with X)': '#c5b0d5',       # Light purple -- KPV fitted WITH the backdoor X
    'OutcomePCLNet (orig)': '#ffbb78',   # Light orange -- shipped closed-form OutcomeNet next to the profiled one (--om_both)
    'DRPCLNET (V1) (orig)': '#aec7e8',   # Light blue   -- DR-V1 built on OutcomePCLNet (orig)
    'DRPCLNET (V2) (orig)': '#f7b6d2',   # Light pink   -- DR-V2 built on OutcomePCLNet (orig) (was #98df8a = DRKPV (Nystrom): collision)

    # BridgeRoutePreference 20x3 study. Every hex below was checked programmatically against the whole
    # palette above and against each other: no collision. Family convention kept (outcome route = blue,
    # treatment route = orange, DR = red, oracles = the desaturated shade of their route, naive = grey).
    'Naive': '#595959',                  # Dark grey -- unadjusted baseline (#7f7f7f is PKDR).
    # Do NOT darken below HLS lightness 0.349 (NMMR-U '#393b79'): plot_pcl_convergence_boxplot hands
    # this WHOLE dict to seaborn, whose line grey is 0.6 * the dict's MINIMUM lightness, so a darker
    # entry silently restyles every boxplot. '#4d4d4d' (L=0.302) did; '#595959' (L=0.349) does not.
    'OutcomeNet': '#0b6fa4',             # Deep blue    -- learned outcome-bridge route
    'TreatmentNet': '#e07b00',           # Deep orange  -- learned treatment-bridge route
    'DRPCLNET': '#b01c1c',               # Deep red     -- doubly robust estimator
    'OracleOM': '#6baed6',               # Light blue   -- exact outcome bridge through the same final stage
    'OracleTM': '#fdae6b',               # Light orange -- exact treatment bridge through the same final stage
    'DR_OracleH': '#756bb1',             # Violet       -- DR with the outcome bridge replaced by its oracle
    'DR_OraclePhi': '#31a354'            # Green        -- DR with the treatment bridge replaced by its oracle
}

def _preprocess_pcl_df(df, x_axis_param):
    """Clean data, rename algorithms, and enforce categorical order."""
    plot_df = df.copy()
    plot_df['Causal_MSE'] = pd.to_numeric(plot_df['Causal_MSE'], errors='coerce')
    plot_df[x_axis_param] = pd.to_numeric(plot_df[x_axis_param], errors='coerce')

    rename_dict = {
        'DRPCLNET_Version1': 'DRPCLNET (V1)',
        'DRPCLNET_Version2': 'DRPCLNET (V2)',
        'OutcomeBridgePCLNET': 'OutcomeNet',
        'TreatmentBridgePCLNET': 'TreatmentNet',

        # DRKPV (Nystrom) variants
        'DRKPV_Nystrom': 'DRKPV (Nystrom)',
        'DRKPV_Nyström': 'DRKPV (Nystrom)',
        'DRKPVNystrom': 'DRKPV (Nystrom)',
        'DRKPV (Nystrom)': 'DRKPV (Nystrom)',
        'DRKPV Nyström': 'DRKPV (Nystrom)',
        'DRKPV_NYS': 'DRKPV (Nystrom)',
        'DRKPV_Nystrom_Approx': 'DRKPV (Nystrom)',
        'KPV_Nystrom': 'KPV (Nystrom)',
        'KAP_Nystrom': 'KAP (Nystrom)',
    }
    plot_df['Algorithm'] = plot_df['Algorithm'].replace(rename_dict)

    # Force the order: Only include items that actually exist in the data
    existing_order = [
        algo for algo in LEGEND_ORDER
        if algo in plot_df['Algorithm'].dropna().unique()
    ]
    plot_df['Algorithm'] = pd.Categorical(
        plot_df['Algorithm'],
        categories=existing_order,
        ordered=True
    )

    return plot_df

def plot_pcl_convergence_ci(df, x_axis_param='Data_Size', title=None, legend_kw=None):
    """Mean and 95% Confidence Interval Line Plot.

    legend_kw: optional dict passed to plt.legend instead of the default (loc='best', ncol=2). Use e.g.
    dict(loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1) to place the legend OUTSIDE the axes when
    'best' would sit on a series (9+ series fill the panel). None = unchanged default behaviour.
    """
    plot_df = _preprocess_pcl_df(df, x_axis_param)

    # 1. Aggregate Mean, Std, and Count to calculate SEM
    summary = (
        plot_df
        .groupby(['Algorithm', x_axis_param], observed=True)['Causal_MSE']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    plt.figure(figsize=(12, 7), dpi=100)
    x_ticks = sorted(summary[x_axis_param].dropna().unique())

    # Iterating through categorical categories ensures the order is respected
    for algo in plot_df['Algorithm'].cat.categories:
        algo_df = summary[summary['Algorithm'] == algo].sort_values(x_axis_param)
        if algo_df.empty:
            continue

        x_vals = algo_df[x_axis_param].values
        mean = algo_df['mean'].values
        std = algo_df['std'].fillna(0.0).values
        n = algo_df['count'].values

        # 2. Calculate 95% Confidence Interval
        sem = std / np.sqrt(n)
        ci_margin = 1.96 * sem

        is_ours = 'DRPCLNET' in str(algo) and '(orig)' not in str(algo)   # the (orig) DR rows are the shipped-OutcomeNet family: dashed, thin

        plt.plot(
            x_vals,
            mean,
            label=algo,
            color=COLOR_PALETTE.get(algo, '#333333'),
            linestyle='-' if is_ours else '--',
            marker='o' if is_ours else 's',
            linewidth=4.0 if is_ours else 2.5,
            markersize=10
        )

        # 3. Fill between Mean +/- 95% CI
        plt.fill_between(
            x_vals,
            np.maximum(mean - ci_margin, 1e-10),
            mean + ci_margin,
            color=COLOR_PALETTE.get(algo, '#333333'),
            alpha=0.15
        )

    # Aesthetics
    plt.yscale('log')
    plt.title(
        title if title else 'Estimator Convergence (Mean with 95% CI)',
        fontsize=20,
        fontweight='bold',
        pad=25
    )
    plt.xlabel('Number of Observations ($N$)', fontsize=18)
    plt.ylabel('Causal Mean Squared Error (Log Scale)', fontsize=18)

    plt.xticks(ticks=x_ticks, labels=[f"{int(x)}" for x in x_ticks], fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    if legend_kw is None:
        plt.legend(fontsize=13, loc='best', frameon=True, shadow=True, ncol=2)
    else:
        plt.legend(**{**dict(fontsize=13, frameon=True, shadow=True), **legend_kw})

    plt.tight_layout()
    return plt

def plot_pcl_convergence_w_quantiles(df, x_axis_param='Data_Size', title=None):
    """Median and IQR Line Plot with enforced legend order."""
    plot_df = _preprocess_pcl_df(df, x_axis_param)
    summary = plot_df.groupby(['Algorithm', x_axis_param], observed=True)['Causal_MSE'].agg(
        median='median', q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    ).reset_index()

    plt.figure(figsize=(12, 7), dpi=100)
    x_ticks = sorted(summary[x_axis_param].unique())

    for algo in plot_df['Algorithm'].cat.categories:
        algo_df = summary[summary['Algorithm'] == algo].sort_values(x_axis_param)
        if algo_df.empty: continue

        x_vals = algo_df[x_axis_param].values
        is_ours = 'DRPCLNET' in algo and '(orig)' not in algo
        plt.plot(x_vals, algo_df['median'], label=algo, color=COLOR_PALETTE.get(algo, '#333333'),
                 linestyle='-' if is_ours else '--', marker='o' if is_ours else 's',
                 linewidth=4.0 if is_ours else 2.5, markersize=10)

        plt.fill_between(x_vals, algo_df['q25'], algo_df['q75'],
                         color=COLOR_PALETTE.get(algo, '#333333'), alpha=0.15)

    plt.yscale('log')
    plt.title(title if title else 'Robust Estimator Convergence (Median & IQR)', fontsize=20, fontweight='bold', pad=25)
    plt.xlabel('Number of Observations ($N$)', fontsize=18)
    plt.ylabel('Causal MSE (Log Scale)', fontsize=18)
    plt.xticks(ticks=x_ticks, labels=[f"{int(x)}" for x in x_ticks], fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(fontsize=13, loc='best', frameon=True, shadow=True, ncol=2)
    plt.tight_layout()
    return plt

def plot_pcl_convergence_boxplot(df, x_axis_param='Data_Size', title=None):
    """Box Plot with enforced categorical legend order."""
    plot_df = _preprocess_pcl_df(df, x_axis_param)
    plt.figure(figsize=(14, 8), dpi=100)

    # Seaborn respects Categorical order for 'hue' automatically
    sns.boxplot(
        data=plot_df.sort_values(x_axis_param),
        x=x_axis_param, y='Causal_MSE', hue='Algorithm',
        palette=COLOR_PALETTE, linewidth=1.5, fliersize=5, showfliers=True
    )

    plt.yscale('log')
    plt.title(title if title else 'Distribution of Estimation Error per Sample Size', fontsize=20, fontweight='bold', pad=25)
    plt.xlabel('Number of Observations ($N$)', fontsize=18)
    plt.ylabel('Causal MSE (Log Scale)', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, axis='y', which="both", linestyle=':', alpha=0.4)
    plt.legend(title='Algorithms', title_fontsize=14, fontsize=12,
               bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, frameon=True, shadow=True)
    plt.tight_layout()
    return plt

def print_experiment_summary_hyperparams(df):
    """
    Summarizes hyperparameters used for each Data_Size in the results dataframe.
    Groups parameters by prefix (out, treat, third) for readability.
    """
    # 1. Identify hyperparameter columns based on your prefixes
    prefixes = ('out_', 'treat_', 'third_')
    hp_cols = [col for col in df.columns if col.startswith(prefixes)]

    # 2. Group by Data_Size and get unique configurations
    # We take the.first because within one Data_Size, HPs are usually constant
    summary_df = df.groupby('Data_Size')[hp_cols].first().reset_index()

    print("="*60)
    print("🚀 EXPERIMENT HYPERPARAMETER SUMMARY")
    print("="*60)

    for _, row in summary_df.iterrows():
        n = row['Data_Size']
        print(f"\n📊 DATA SIZE: N = {n}")
        print("-" * 30)

        # Grouping by prefix for cleaner output
        for prefix in prefixes:
            section_name = {
                'out_': 'Outcome Bridge (First/Second Stage)',
                'treat_': 'Treatment Bridge (First/Second Stage)',
                'third_': 'Third Stage (Inference/Slack)'
            }[prefix]

            print(f"\n  🔹 {section_name}:")
            current_prefix_cols = [c for c in hp_cols if c.startswith(prefix)]

            for col in current_prefix_cols:
                # Clean the name for printing (e.g., out_lr -> lr)
                display_name = col.replace(prefix, "")
                value = row[col]
                print(f"    {display_name:<25}: {value}")

        print("\n" + "."*60)