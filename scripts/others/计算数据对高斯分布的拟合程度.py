import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, chi2
from scipy.spatial.transform import Rotation
import os
import sys


def fit_gaussian_and_evaluate(csv_file_path, output_dir="gaussian_fit_results",
                              bin_width=None, num_bins=None):
    """
    Read CSV odometry data, convert quaternions to Euler angles (rad/deg),
    fit Gaussian distribution to all columns, and evaluate goodness of fit

    Parameters:
        csv_file_path: Path to CSV file
        output_dir: Directory to save results
        bin_width: Fixed width of each histogram bin (overrides num_bins if specified)
        num_bins: Number of histogram bins (used if bin_width is None)
                  If both are None, uses Sturges' rule to calculate optimal bins
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read CSV file
    print(f"Reading file: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        sys.exit(1)

    # Convert quaternions to Euler angles (ROS standard RPY order: XYZ)
    print("Converting quaternions to Euler angles...")
    required_quat_cols = ['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
    if all(col in df.columns for col in required_quat_cols):
        # Extract quaternions (x, y, z, w)
        quats = df[required_quat_cols].values

        # Normalize quaternions to ensure valid rotation
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # Create rotation objects
        rotations = Rotation.from_quat(quats)

        # Convert to Euler angles (radians and degrees)
        euler_rad = rotations.as_euler('xyz', degrees=False)
        euler_deg = rotations.as_euler('xyz', degrees=True)

        # Add to DataFrame
        df['roll_rad'] = euler_rad[:, 0]  # Rotation around X axis (radians)
        df['pitch_rad'] = euler_rad[:, 1]  # Rotation around Y axis (radians)
        df['yaw_rad'] = euler_rad[:, 2]  # Rotation around Z axis (radians)
        df['roll_deg'] = euler_deg[:, 0]  # Rotation around X axis (degrees)
        df['pitch_deg'] = euler_deg[:, 1]  # Rotation around Y axis (degrees)
        df['yaw_deg'] = euler_deg[:, 2]  # Rotation around Z axis (degrees)

        print("Successfully converted quaternions to Euler angles (RPY order)")
    else:
        print("Warning: Quaternion columns not found, skipping Euler angle conversion")

    # Get numeric columns (including new Euler angle columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Detected numeric columns: {numeric_cols}")
    print(f"Total data rows: {len(df)}")

    # Initialize results dictionary
    results = {}

    # Process each column
    for col in numeric_cols:
        data = df[col].dropna().values
        n = len(data)

        if n < 5:
            print(f"Warning: Insufficient valid data for column {col}, skipping")
            continue

        # 1. Gaussian distribution fitting (Maximum Likelihood Estimation)
        mu, std = norm.fit(data)

        # 2. Calculate descriptive statistics
        mean = np.mean(data)
        median = np.median(data)
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)  # Excess kurtosis (normal distribution = 0)

        # 3. Determine histogram bins (核心修改部分)
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min

        if bin_width is not None:
            # 方法1: 指定固定bin宽度
            num_bins_calc = int(np.ceil(data_range / bin_width))
            # 确保至少有1个bin
            num_bins_calc = max(1, num_bins_calc)
            # 生成等宽的bin边界
            bin_edges = np.linspace(data_min, data_min + num_bins_calc * bin_width, num_bins_calc + 1)
            print(f"Column {col}: Using fixed bin width={bin_width}, total bins={num_bins_calc}")
        elif num_bins is not None:
            # 方法2: 指定bin数量
            num_bins_calc = num_bins
            bin_edges = np.linspace(data_min, data_max, num_bins_calc + 1)
            print(f"Column {col}: Using specified number of bins={num_bins_calc}")
        else:
            # 方法3: 默认使用Sturges'规则
            num_bins_calc = int(np.ceil(np.log2(n) + 1))
            bin_edges = np.linspace(data_min, data_max, num_bins_calc + 1)
            print(f"Column {col}: Using Sturges' rule, bins={num_bins_calc}")

        # 4. Goodness-of-fit tests
        # Shapiro-Wilk test (best for small samples, n < 2000)
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, std))

        # Chi-square test (使用上面计算的bin_edges)
        hist, _ = np.histogram(data, bins=bin_edges)

        # Calculate theoretical frequencies for each bin
        expected = []
        for i in range(num_bins_calc):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            prob = norm.cdf(upper, mu, std) - norm.cdf(lower, mu, std)
            expected.append(prob * n)

        # Merge bins with expected frequency < 5 (required for chi-square test)
        observed = []
        expected_merged = []
        current_obs = 0
        current_exp = 0

        for obs, exp in zip(hist, expected):
            current_obs += obs
            current_exp += exp
            if current_exp >= 5:
                observed.append(current_obs)
                expected_merged.append(current_exp)
                current_obs = 0
                current_exp = 0

        if current_obs > 0 or current_exp > 0:
            if observed:
                observed[-1] += current_obs
                expected_merged[-1] += current_exp
            else:
                observed.append(current_obs)
                expected_merged.append(current_exp)

        # Calculate chi-square statistic and p-value
        chi2_stat = np.sum((np.array(observed) - np.array(expected_merged)) ** 2 / np.array(expected_merged))
        df_chi2 = len(observed) - 2 - 1  # Degrees of freedom = bins - estimated parameters - 1
        chi2_p = 1 - chi2.cdf(chi2_stat, df_chi2) if df_chi2 > 0 else np.nan

        # 5. Save results
        results[col] = {
            'sample_size': n,
            'mean': mean,
            'median': median,
            'std_dev': std,
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'chi2_stat': chi2_stat,
            'chi2_df': df_chi2,
            'chi2_p': chi2_p,
            'num_bins': num_bins_calc,
            'bin_width': bin_width if bin_width is not None else data_range / num_bins_calc
        }

        # 6. Plot histogram and fitted curve (使用相同的bin_edges)
        plt.figure(figsize=(10, 6))
        # 关键：使用bins=bin_edges而不是num_bins，确保宽度完全一致
        plt.hist(data, bins=bin_edges, density=True, alpha=0.6, color='skyblue',
                 label='Actual Data Distribution', edgecolor='white', linewidth=0.5)

        # Plot fitted Gaussian curve
        x = np.linspace(data_min, data_max, 1000)
        y = norm.pdf(x, mu, std)
        plt.plot(x, y, 'r-', linewidth=2, label=f'Gaussian Fit (μ={mu:.6f}, σ={std:.6f})')

        plt.title(f'{col} Data Distribution vs Gaussian Fit')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_gaussian_fit.png'), dpi=300)
        plt.close()

    # 7. Generate detailed report
    report_path = os.path.join(output_dir, 'gaussian_fit_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ODOMETRY DATA GAUSSIAN DISTRIBUTION FIT REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data file: {csv_file_path}\n")
        f.write(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if all(col in df.columns for col in required_quat_cols):
            f.write("Quaternion conversion: Enabled (RPY/XYZ order, radians + degrees)\n")
        f.write("\n")

        for col, res in results.items():
            f.write("-" * 80 + "\n")
            f.write(f"Column: {col}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sample size: {res['sample_size']}\n")
            f.write(f"Number of bins: {res['num_bins']}\n")
            f.write(f"Average bin width: {res['bin_width']:.10f}\n\n")

            f.write("Descriptive Statistics:\n")
            f.write(f"  Mean: {res['mean']:.10f}\n")
            f.write(f"  Median: {res['median']:.10f}\n")
            f.write(f"  Standard deviation: {res['std_dev']:.10f}\n")
            f.write(f"  Skewness: {res['skewness']:.6f} (normal = 0)\n")
            f.write(f"  Excess kurtosis: {res['excess_kurtosis']:.6f} (normal = 0)\n\n")

            f.write("Goodness-of-Fit Tests (p-value > 0.05 means cannot reject normality):\n")
            f.write(f"  Shapiro-Wilk test: statistic={res['shapiro_stat']:.6f}, p-value={res['shapiro_p']:.6f}\n")
            f.write(f"  Kolmogorov-Smirnov test: statistic={res['ks_stat']:.6f}, p-value={res['ks_p']:.6f}\n")
            if not np.isnan(res['chi2_p']):
                f.write(
                    f"  Chi-square test: statistic={res['chi2_stat']:.6f}, df={res['chi2_df']}, p-value={res['chi2_p']:.6f}\n")
            else:
                f.write(f"  Chi-square test: Cannot calculate (insufficient degrees of freedom)\n")

            # Overall assessment
            p_values = [res['shapiro_p'], res['ks_p']]
            if not np.isnan(res['chi2_p']):
                p_values.append(res['chi2_p'])

            avg_p = np.mean(p_values)
            f.write("\nOverall Fit Assessment:\n")
            if avg_p > 0.1:
                f.write("  Fit level: Excellent (average p-value > 0.1)\n")
            elif avg_p > 0.05:
                f.write("  Fit level: Good (average p-value > 0.05)\n")
            elif avg_p > 0.01:
                f.write("  Fit level: Fair (average p-value > 0.01)\n")
            else:
                f.write("  Fit level: Poor (average p-value ≤ 0.01)\n")

            f.write("\n")

    # 8. Print results summary
    print("\n" + "=" * 80)
    print("FIT RESULTS SUMMARY")
    print("=" * 80)
    print(f"Results saved to directory: {output_dir}")
    print("\nFit levels for each column:")
    for col, res in results.items():
        p_values = [res['shapiro_p'], res['ks_p']]
        if not np.isnan(res['chi2_p']):
            p_values.append(res['chi2_p'])
        avg_p = np.mean(p_values)

        if avg_p > 0.1:
            fit_level = "Excellent"
        elif avg_p > 0.05:
            fit_level = "Good"
        elif avg_p > 0.01:
            fit_level = "Fair"
        else:
            fit_level = "Poor"

        print(f"  {col}: {fit_level} (average p-value={avg_p:.4f}, bins={res['num_bins']})")

    return results


if __name__ == "__main__":
    csv_file = r"/home/awwsome/odom_data/静止1 _point_raw.csv"

    # 使用示例1: 指定固定bin宽度（推荐用于里程计数据）
    # 例如，对于角度数据(度)，可以设为0.01度；对于弧度数据，设为0.0001弧度
    results = fit_gaussian_and_evaluate(csv_file, bin_width=0.001)

    # 使用示例2: 指定bin数量
    # results = fit_gaussian_and_evaluate(csv_file, num_bins=50)

    # 使用示例3: 使用默认的Sturges'规则
    # results = fit_gaussian_and_evaluate(csv_file)