#!/usr/bin/env python3
"""
Statistical Analysis Tool for MIR Features

Scans .INFO files recursively and calculates statistics for each feature:
- Count, mean, median, std, min, max for numeric features
- Value distribution for categorical features
- Outlier detection using IQR method

Usage:
    python src/tools/statistical_analysis.py /path/to/data
    python src/tools/statistical_analysis.py /path/to/data --output stats.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.common import setup_logging

logger = logging.getLogger(__name__)

# Feature units reference for output display
FEATURE_UNITS = {
    # Rhythm
    "bpm": "BPM",
    "beat_count": "count",
    "beat_regularity": "sec",
    "bpm_is_defined": "bool",
    "onset_count": "count",
    "onset_density": "/sec",
    "onset_strength_mean": "-",
    "onset_strength_std": "-",
    "syncopation_index": "0-1",
    "rhythmic_complexity": "0-1",
    # Loudness
    "lufs": "LUFS",
    "lra": "LU",
    "lufs_bass": "LUFS",
    "lufs_drums": "LUFS",
    "lufs_other": "LUFS",
    "lufs_vocals": "LUFS",
    "lra_bass": "LU",
    "lra_drums": "LU",
    "lra_other": "LU",
    "lra_vocals": "LU",
    # Spectral
    "spectral_flatness": "0-1",
    "spectral_flux": "-",
    "spectral_skewness": "-",
    "spectral_kurtosis": "-",
    # RMS Energy
    "rms_bass": "dB",
    "rms_body": "dB",
    "rms_mid": "dB",
    "rms_air": "dB",
    # Timbral (Audio Commons)
    "brightness": "0-100",
    "roughness": "0-100",
    "hardness": "0-100",
    "depth": "0-100",
    "booming": "0-100",
    "reverberation": "0-100",
    "sharpness": "0-100",
    "warmth": "0-100",
    # Classification
    "danceability": "0-1",
    "atonality": "0-1",
    # AudioBox Aesthetics
    "audiobox_content_enjoyment": "1-10",
    "audiobox_content_usefulness": "1-10",
    "audiobox_production_complexity": "1-10",
    "audiobox_production_quality": "1-10",
    # Chroma (all 0-1)
    "chroma_C": "0-1", "chroma_C#": "0-1", "chroma_D": "0-1", "chroma_D#": "0-1",
    "chroma_E": "0-1", "chroma_F": "0-1", "chroma_F#": "0-1", "chroma_G": "0-1",
    "chroma_G#": "0-1", "chroma_A": "0-1", "chroma_A#": "0-1", "chroma_B": "0-1",
}


def get_unit(feature_name: str) -> str:
    """Get the unit for a feature, or '-' if unknown."""
    return FEATURE_UNITS.get(feature_name, "-")


def find_info_files(root_path: Path) -> List[Path]:
    """Find all .INFO files recursively."""
    info_files = list(root_path.rglob("*.INFO"))
    logger.info(f"Found {len(info_files)} .INFO files")
    return info_files


def load_all_features(info_files: List[Path]) -> tuple:
    """
    Load all features from .INFO files into a dict of lists.
    
    Returns:
        Tuple of (features dict, file_names list)
    """
    features: Dict[str, List[Any]] = defaultdict(list)
    file_names: List[str] = []
    
    for info_file in info_files:
        try:
            with open(info_file) as f:
                data = json.load(f)
            
            # Track filename (use parent folder name)
            file_names.append(info_file.parent.name)
            
            for key, value in data.items():
                features[key].append(value)
                
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {info_file}: {e}")
            
    return dict(features), file_names


def calculate_numeric_stats(values: List[float], file_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate statistics for numeric values.
    
    Args:
        values: List of numeric values
        file_names: Optional list of filenames corresponding to values
    """
    # Create indices for valid numeric values
    valid_indices = [i for i, v in enumerate(values) 
                     if isinstance(v, (int, float)) and v is not None]
    numeric_values = [values[i] for i in valid_indices]
    
    if not numeric_values:
        return {"count": 0, "error": "no numeric values"}
    
    arr = np.array(numeric_values)
    
    # Calculate quartiles for IQR
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (arr < lower_bound) | (arr > upper_bound)
    outlier_count = int(np.sum(outlier_mask))
    
    # Get outlier filenames if available
    outlier_files = []
    if file_names and outlier_count > 0:
        for idx, is_outlier in enumerate(outlier_mask):
            if is_outlier:
                original_idx = valid_indices[idx]
                if original_idx < len(file_names):
                    outlier_files.append({
                        "file": file_names[original_idx],
                        "value": float(arr[idx])
                    })
    
    return {
        "count": len(numeric_values),
        "mean": float(np.mean(arr)),
        "median": float(median),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "outliers": outlier_count,
        "outlier_pct": float(outlier_count / len(numeric_values) * 100) if numeric_values else 0,
        "outlier_files": outlier_files
    }


def calculate_categorical_stats(values: List[Any]) -> Dict[str, Any]:
    """Calculate statistics for categorical/string values."""
    # Count occurrences
    value_counts: Dict[str, int] = defaultdict(int)
    
    for v in values:
        if isinstance(v, dict):
            # For dict values (like essentia_genre), count top keys
            for key, score in v.items():
                if isinstance(score, (int, float)) and score > 0.5:
                    value_counts[key] += 1
        elif isinstance(v, str):
            value_counts[v] += 1
        elif isinstance(v, list):
            for item in v:
                value_counts[str(item)] += 1
                
    # Sort by count
    sorted_counts = dict(sorted(value_counts.items(), key=lambda x: -x[1]))
    
    return {
        "count": len(values),
        "unique_values": len(sorted_counts),
        "top_10": dict(list(sorted_counts.items())[:10]),
        "distribution": sorted_counts
    }


def is_numeric_feature(values: List[Any]) -> bool:
    """Check if a feature is numeric based on its values."""
    for v in values[:10]:  # Check first 10 values
        if v is not None and not isinstance(v, (int, float)):
            return False
    return True


def analyze_features(features: Dict[str, List[Any]], file_names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Analyze all features and return statistics."""
    stats = {}
    
    for key, values in sorted(features.items()):
        if not values:
            continue
            
        # Skip text-heavy features (Music Flamingo descriptions)
        if key.startswith("music_flamingo_") and "full" in key or len(str(values[0])) > 200:
            stats[key] = {
                "type": "text",
                "count": len(values),
                "avg_length": np.mean([len(str(v)) for v in values if v])
            }
            continue
            
        # Check if numeric or categorical
        if is_numeric_feature(values):
            stats[key] = {
                "type": "numeric",
                **calculate_numeric_stats(values, file_names)
            }
        else:
            stats[key] = {
                "type": "categorical",
                **calculate_categorical_stats(values)
            }
            
    return stats


def print_summary(stats: Dict[str, Dict], verbose: bool = False):
    """Print a human-readable summary with feature units."""
    print("\n" + "=" * 90)
    print("FEATURE STATISTICS SUMMARY")
    print("=" * 90)
    
    # Group by type
    numeric_features = {k: v for k, v in stats.items() if v.get("type") == "numeric"}
    categorical_features = {k: v for k, v in stats.items() if v.get("type") == "categorical"}
    text_features = {k: v for k, v in stats.items() if v.get("type") == "text"}
    
    print(f"\nTotal Features: {len(stats)}")
    print(f"  Numeric: {len(numeric_features)}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Text: {len(text_features)}")
    
    # Numeric features table with units
    if numeric_features:
        print("\n" + "-" * 90)
        print("NUMERIC FEATURES")
        print("-" * 90)
        print(f"{'Feature':<30} {'Unit':<8} {'Count':>5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 90)
        
        for key, s in sorted(numeric_features.items()):
            if "error" in s:
                continue
            unit = get_unit(key)
            print(f"{key:<30} {unit:<8} {s['count']:>5} {s['mean']:>10.3f} {s['std']:>10.3f} {s['min']:>10.3f} {s['max']:>10.3f}")
            
            if verbose and s.get("outliers", 0) > 0:
                print(f"  └── Outliers: {s['outliers']} ({s['outlier_pct']:.1f}%)")
                # Show outlier files
                for outlier in s.get("outlier_files", []):
                    print(f"      • {outlier['file']} (value: {outlier['value']:.3f})")
    
    # Categorical features
    if categorical_features and verbose:
        print("\n" + "-" * 70)
        print("CATEGORICAL FEATURES (Top Values)")
        print("-" * 70)
        
        for key, s in sorted(categorical_features.items()):
            print(f"\n{key}: {s['count']} samples, {s['unique_values']} unique values")
            for value, count in list(s.get("top_10", {}).items())[:5]:
                print(f"  - {value}: {count}")
    
    print("\n" + "=" * 70)


def calculate_correlation_matrix(features: Dict[str, List[Any]]) -> tuple:
    """
    Calculate correlation matrix for all numeric features.
    
    Returns:
        Tuple of (correlation_matrix, feature_names, data_matrix)
    """
    # Get numeric features only
    numeric_features = {}
    for key, values in features.items():
        if is_numeric_feature(values):
            numeric_values = [v if isinstance(v, (int, float)) and v is not None else np.nan 
                            for v in values]
            if len(numeric_values) > 0:
                numeric_features[key] = numeric_values
    
    if len(numeric_features) < 2:
        logger.warning("Need at least 2 numeric features for correlation analysis")
        return None, [], None
    
    # Convert to matrix (features x samples)
    feature_names = sorted(numeric_features.keys())
    n_samples = len(list(numeric_features.values())[0])
    
    # Build data matrix
    data_matrix = np.zeros((len(feature_names), n_samples))
    for i, name in enumerate(feature_names):
        values = numeric_features[name]
        # Pad or truncate to match n_samples
        if len(values) >= n_samples:
            data_matrix[i, :] = values[:n_samples]
        else:
            data_matrix[i, :len(values)] = values
            data_matrix[i, len(values):] = np.nan
    
    # Calculate correlation matrix (handles NaN)
    corr_matrix = np.zeros((len(feature_names), len(feature_names)))
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            # Get valid pairs (both not NaN)
            mask = ~np.isnan(data_matrix[i]) & ~np.isnan(data_matrix[j])
            if np.sum(mask) > 2:
                x = data_matrix[i, mask]
                y = data_matrix[j, mask]
                # Standardize
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0
                else:
                    corr_matrix[i, j] = 0
            else:
                corr_matrix[i, j] = 0
    
    return corr_matrix, feature_names, data_matrix


def calculate_max_correlation(corr_matrix: np.ndarray, feature_names: List[str]) -> Dict[str, Dict]:
    """
    For each feature, calculate max absolute correlation with any other feature.
    
    Returns:
        Dict mapping feature name to {max_corr, most_correlated_with}
    """
    results = {}
    n = len(feature_names)
    
    for i, name in enumerate(feature_names):
        # Get correlations with all other features (exclude self)
        correlations = []
        for j in range(n):
            if i != j:
                correlations.append((abs(corr_matrix[i, j]), corr_matrix[i, j], feature_names[j]))
        
        if correlations:
            # Sort by absolute correlation
            correlations.sort(key=lambda x: -x[0])
            max_abs, max_val, most_corr_with = correlations[0]
            
            # Also calculate mean correlation with all other features
            mean_corr = np.mean([abs(c[1]) for c in correlations])
            
            results[name] = {
                "max_abs_correlation": float(max_abs),
                "max_correlation_value": float(max_val),
                "most_correlated_with": most_corr_with,
                "mean_abs_correlation": float(mean_corr)
            }
    
    return results


def print_correlation_summary(corr_results: Dict[str, Dict], threshold: float = 0.7):
    """Print correlation analysis summary."""
    print("\n" + "=" * 95)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 95)
    print("(Max absolute correlation with any other feature - higher = more redundant)")
    print("-" * 95)
    print(f"{'Feature':<35} {'MaxCorr':>8} {'MeanCorr':>9} {'Most Correlated With':<35}")
    print("-" * 95)
    
    # Sort by max correlation (descending)
    sorted_results = sorted(corr_results.items(), key=lambda x: -x[1]["max_abs_correlation"])
    
    high_corr_pairs = []
    
    for name, data in sorted_results:
        max_corr = data["max_abs_correlation"]
        mean_corr = data["mean_abs_correlation"]
        most_corr = data["most_correlated_with"]
        
        # Flag high correlations
        flag = "⚠️ " if max_corr >= threshold else "   "
        print(f"{flag}{name:<32} {max_corr:>8.3f} {mean_corr:>9.3f} {most_corr:<35}")
        
        if max_corr >= threshold:
            # Track unique pairs
            pair = tuple(sorted([name, most_corr]))
            if pair not in high_corr_pairs:
                high_corr_pairs.append(pair)
    
    print("-" * 95)
    
    # Summary of highly correlated pairs
    if high_corr_pairs:
        print(f"\n⚠️  HIGHLY CORRELATED PAIRS (|r| >= {threshold}):")
        print("    Consider removing one from each pair to reduce redundancy:\n")
        for pair in high_corr_pairs:
            # Get the actual correlation value
            if pair[0] in corr_results:
                if corr_results[pair[0]]["most_correlated_with"] == pair[1]:
                    corr_val = corr_results[pair[0]]["max_correlation_value"]
                else:
                    corr_val = corr_results[pair[1]]["max_correlation_value"]
            else:
                corr_val = corr_results[pair[1]]["max_correlation_value"]
            print(f"    • {pair[0]} ↔ {pair[1]} (r = {corr_val:.3f})")
    else:
        print(f"\n✅ No highly correlated pairs found (threshold: |r| >= {threshold})")
    
    # Qualitative legend
    print("\n" + "-" * 95)
    print("CORRELATION INTERPRETATION GUIDE:")
    print("-" * 95)
    print("  |r| = 0.0 - 0.3  : Weak correlation      → Features are largely independent")
    print("  |r| = 0.3 - 0.5  : Moderate correlation  → Some shared information, both useful")
    print("  |r| = 0.5 - 0.7  : Strong correlation    → Consider if both are needed")
    print("  |r| = 0.7 - 0.9  : Very strong           → Likely redundant, remove one")
    print("  |r| = 0.9 - 1.0  : Near identical        → Definitely remove one")
    print("-" * 95)
    print("  Positive r: Features increase together (e.g., brightness ↔ sharpness)")
    print("  Negative r: Features are inversely related (e.g., depth ↔ brightness)")
    print("=" * 95)


def print_legend():
    """Print explanation of all statistical variables."""
    print("=" * 80)
    print("STATISTICAL ANALYSIS LEGEND")
    print("=" * 80)
    
    print("\n📊 BASIC STATISTICS:")
    print("-" * 80)
    print("  count       : Number of files containing this feature")
    print("  mean        : Average value (sum / count)")
    print("  median      : Middle value when sorted (50th percentile)")
    print("                → Less affected by outliers than mean")
    print("  std         : Standard deviation - spread around the mean")
    print("                → Low std = consistent values, High std = variable")
    print("  min / max   : Smallest and largest values observed")
    
    print("\n📦 QUARTILES & OUTLIERS:")
    print("-" * 80)
    print("  q1          : First quartile (25th percentile)")
    print("  q3          : Third quartile (75th percentile)")
    print("  iqr         : Interquartile range = q3 - q1")
    print("                → Robust measure of spread (ignores extremes)")
    print("  outliers    : Count of values outside 1.5×IQR from quartiles")
    print("  outlier_pct : Percentage of values that are outliers")
    print("  outlier_files: List of files with outlier values")
    
    print("\n🔗 CORRELATION ANALYSIS (with --correlation flag):")
    print("-" * 80)
    print("  max_abs_correlation   : Highest |r| with any other feature (0-1)")
    print("  max_correlation_value : Actual r value (-1 to +1)")
    print("  most_correlated_with  : Feature name with highest correlation")
    print("  mean_abs_correlation  : Average |r| across all other features")
    
    print("\n📈 CORRELATION INTERPRETATION:")
    print("-" * 80)
    print("  |r| = 0.0 - 0.3 : Weak         → Features are independent")
    print("  |r| = 0.3 - 0.5 : Moderate     → Some shared information")
    print("  |r| = 0.5 - 0.7 : Strong       → Consider if both needed")
    print("  |r| = 0.7 - 0.9 : Very strong  → Likely redundant, remove one")
    print("  |r| = 0.9 - 1.0 : Near identical → Definitely remove one")
    print()
    print("  Positive r: Features increase together")
    print("  Negative r: Features are inversely related")
    
    print("\n💡 USAGE TIPS:")
    print("-" * 80)
    print("  • High outlier_pct may indicate data quality issues")
    print("  • If mean ≠ median, distribution is skewed (check outliers)")
    print("  • Features with |r| > 0.7 may confuse ML models - pick one")
    print("  • Low std relative to mean = consistent across your dataset")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature statistics across .INFO files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a folder
  python statistical_analysis.py /path/to/data
  
  # Save results to JSON
  python statistical_analysis.py /path/to/data --output stats.json
  
  # Verbose output with outlier details
  python statistical_analysis.py /path/to/data --verbose
  
  # Include correlation analysis
  python statistical_analysis.py /path/to/data --correlation
  
  # Set correlation threshold (default 0.7)
  python statistical_analysis.py /path/to/data --correlation --corr-threshold 0.8
  
  # Show explanation of all statistical variables
  python statistical_analysis.py --legend
        """
    )
    
    parser.add_argument("path", nargs='?', help="Root directory to scan for .INFO files")
    parser.add_argument("--output", "-o", help="Output JSON file for statistics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--correlation", "-c", action="store_true", 
                        help="Calculate correlation between features")
    parser.add_argument("--corr-threshold", type=float, default=0.7,
                        help="Threshold for flagging high correlations (default: 0.7)")
    parser.add_argument("--legend", "-l", action="store_true",
                        help="Print explanation of statistical variables and exit")
    
    args = parser.parse_args()
    
    # Handle --legend flag (no path required)
    if args.legend:
        print_legend()
        return None
    
    # Path is required if not using --legend
    if not args.path:
        parser.error("path is required unless using --legend")
    
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    root_path = Path(args.path)
    
    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        sys.exit(1)
        
    # Find and load all .INFO files
    info_files = find_info_files(root_path)
    
    if not info_files:
        logger.error("No .INFO files found")
        sys.exit(1)
        
    # Load features
    logger.info("Loading features from .INFO files...")
    features, file_names = load_all_features(info_files)
    logger.info(f"Found {len(features)} unique feature keys")
    
    # Calculate statistics
    logger.info("Calculating statistics...")
    stats = analyze_features(features, file_names)
    
    # Print summary
    print_summary(stats, verbose=args.verbose)
    
    # Correlation analysis
    if args.correlation:
        logger.info("Calculating feature correlations...")
        corr_matrix, feature_names, _ = calculate_correlation_matrix(features)
        
        if corr_matrix is not None:
            corr_results = calculate_max_correlation(corr_matrix, feature_names)
            print_correlation_summary(corr_results, threshold=args.corr_threshold)
            
            # Add to stats for JSON output
            stats["_correlation_analysis"] = corr_results
    
    # Add legend to JSON output
    stats["_legend"] = {
        "statistics": {
            "count": "Number of files with this feature",
            "mean": "Average value across all files",
            "median": "Middle value (50th percentile) - less affected by outliers than mean",
            "std": "Standard deviation - spread of values around the mean",
            "min": "Minimum value observed",
            "max": "Maximum value observed",
            "q1": "First quartile (25th percentile)",
            "q3": "Third quartile (75th percentile)",
            "iqr": "Interquartile range (q3 - q1) - robust measure of spread",
            "outliers": "Count of values outside 1.5*IQR from quartiles",
            "outlier_pct": "Percentage of values that are outliers",
            "outlier_files": "List of files with outlier values for this feature"
        },
        "correlation": {
            "max_abs_correlation": "Highest absolute correlation with any other feature (0-1)",
            "max_correlation_value": "Actual correlation value (-1 to 1, can be negative)",
            "most_correlated_with": "Feature name with highest correlation",
            "mean_abs_correlation": "Average absolute correlation with all other features"
        },
        "interpretation": {
            "low_std": "Low std relative to mean suggests consistent values across tracks",
            "high_outliers": "High outlier_pct may indicate data quality issues or natural variation",
            "correlation_threshold": "Features with |r| > 0.7 may be redundant for ML training",
            "correlation_ranges": {
                "0.0-0.3": "Weak - features are independent",
                "0.3-0.5": "Moderate - some shared information",
                "0.5-0.7": "Strong - consider if both needed",
                "0.7-0.9": "Very strong - likely redundant",
                "0.9-1.0": "Near identical - definitely remove one"
            }
        }
    }
    
    # Save to JSON
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved to: {output_path}")
    
    return stats


if __name__ == "__main__":
    main()
