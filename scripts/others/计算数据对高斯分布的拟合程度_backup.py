import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, chi2
from scipy.spatial.transform import Rotation
import os
import sys


def fit_gaussian_and_evaluate(csv_file_path, output_dir="gaussian_fit_results"):
    """
    Read CSV odometry data, convert quaternions to Euler angles (rad/deg),
    fit Gaussian distribution to all columns, and evaluate goodness of fit

    Parameters:
        csv_file_path: Path to CSV file
        output_dir: Directory to save results
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

        # 3. Goodness-of-fit tests
        # Shapiro-Wilk test (best for small samples, n < 2000)
        shapiro_stat, shapiro_p = stats.shapiro(data)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, std))

        # Chi-square test
        # Determine number of bins (Sturges' rule)
        num_bins = int(np.ceil(np.log2(n) + 1))
        hist, bin_edges = np.histogram(data, bins=num_bins)

        # Calculate theoretical frequencies for each bin
        expected = []
        for i in range(num_bins):
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

        # 4. Save results
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
            'chi2_p': chi2_p
        }

        # 5. Plot histogram and fitted curve
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=num_bins, density=True, alpha=0.6, color='skyblue', label='Actual Data Distribution')

        # Plot fitted Gaussian curve
        x = np.linspace(data.min(), data.max(), 1000)
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

    # 6. Generate detailed report
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
            f.write(f"Sample size: {res['sample_size']}\n\n")

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

    # 7. Print results summary
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

        print(f"  {col}: {fit_level} (average p-value={avg_p:.4f})")

    return results


if __name__ == "__main__":
    csv_file = r"/home/awwsome/odom_data/静止1 _point_raw.csv"

    # Run analysis
    results = fit_gaussian_and_evaluate(csv_file)