import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.spatial.transform import Rotation
import os
import sys


def read_and_process_csv(csv_file_path):
    """
    Read CSV odometry file and convert quaternions to Euler angles (rad/deg)

    Parameters:
        csv_file_path: Path to CSV file

    Returns:
        DataFrame with processed data (including Euler angles)
    """
    print(f"Processing file: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        sys.exit(1)

    # Convert quaternions to Euler angles (ROS standard RPY order: XYZ)
    required_quat_cols = ['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']
    if all(col in df.columns for col in required_quat_cols):
        # Extract and normalize quaternions
        quats = df[required_quat_cols].values
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

        # Convert to Euler angles
        rotations = Rotation.from_quat(quats)
        euler_rad = rotations.as_euler('xyz', degrees=False)
        euler_deg = rotations.as_euler('xyz', degrees=True)

        # Add to DataFrame
        df['roll_rad'] = euler_rad[:, 0]
        df['pitch_rad'] = euler_rad[:, 1]
        df['yaw_rad'] = euler_rad[:, 2]
        df['roll_deg'] = euler_deg[:, 0]
        df['pitch_deg'] = euler_deg[:, 1]
        df['yaw_deg'] = euler_deg[:, 2]

        print("  ✓ Quaternions converted to Euler angles (RPY order)")

    return df


def compare_distributions(csv_file1, csv_file2, output_dir="distribution_comparison"):
    """
    Compare probability density functions of two odometry datasets

    Parameters:
        csv_file1: Path to first CSV file
        csv_file2: Path to second CSV file
        output_dir: Directory to save comparison plots
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process both files
    df1 = read_and_process_csv(csv_file1)
    df2 = read_and_process_csv(csv_file2)

    # Find common numeric columns
    numeric_cols1 = df1.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
    common_cols = sorted(list(set(numeric_cols1) & set(numeric_cols2)))

    if not common_cols:
        print("\nError: No common numeric columns found between the two files.")
        sys.exit(1)

    print(f"\nFound {len(common_cols)} common columns for comparison:")
    for col in common_cols:
        print(f"  - {col}")

    # Get file names for legend
    file1_name = os.path.basename(csv_file1)
    file2_name = os.path.basename(csv_file2)

    # Process each common column
    for col in common_cols:
        data1 = df1[col].dropna().values
        data2 = df2[col].dropna().values

        if len(data1) < 10 or len(data2) < 10:
            print(f"\nWarning: Insufficient data for column '{col}', skipping")
            continue

        print(f"\nGenerating plot for: {col}")

        # Calculate Kernel Density Estimation (KDE) for smooth PDF
        kde1 = gaussian_kde(data1, bw_method='scott')
        kde2 = gaussian_kde(data2, bw_method='scott')

        # Determine x-axis range (cover both datasets with some padding)
        x_min = min(data1.min(), data2.min())
        x_max = max(data1.max(), data2.max())
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)

        # Calculate PDF values
        pdf1 = kde1(x)
        pdf2 = kde2(x)

        # Calculate standard normal distribution (μ=0, σ=1)
        x_std = np.linspace(-3, 3, 1000)
        pdf_std = norm.pdf(x_std, 0, 1)

        # Create plot
        plt.figure(figsize=(12, 7))

        # Plot datasets
        plt.plot(x, pdf1, 'b-', linewidth=2.5, alpha=0.8, label=f'{file1_name} (n={len(data1)})')
        plt.plot(x, pdf2, 'r-', linewidth=2.5, alpha=0.8, label=f'{file2_name} (n={len(data2)})')

        # Plot standard normal distribution
        plt.plot(x_std, pdf_std, 'k--', linewidth=2, alpha=0.7, label='Standard Normal (μ=0, σ=1)')

        # Add statistics annotations
        stats_text = (
            f"{file1_name}:\n"
            f"  Mean: {np.mean(data1):.6f}\n"
            f"  Std: {np.std(data1):.6f}\n\n"
            f"{file2_name}:\n"
            f"  Mean: {np.mean(data2):.6f}\n"
            f"  Std: {np.std(data2):.6f}"
        )
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title(f'Probability Density Function Comparison\n{col}', fontsize=14, pad=20)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(output_dir, f'{col}_pdf_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Generate summary report
    report_path = os.path.join(output_dir, 'comparison_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ODOMETRY DISTRIBUTION COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"File 1: {csv_file1}\n")
        f.write(f"File 2: {csv_file2}\n")
        f.write(f"Analysis time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Common columns compared: {len(common_cols)}\n")
        for col in common_cols:
            f.write(f"  - {col}\n")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"All plots and summary saved to: {output_dir}")
    print(f"Summary report: {report_path}")


if __name__ == "__main__":
    csv1 = r"/home/awwsome/odom_data/静止1_fast_raw.csv"
    csv2 = r"/home/awwsome/odom_data/静止1 _point_raw.csv"

    # Run comparison
    compare_distributions(csv1, csv2)