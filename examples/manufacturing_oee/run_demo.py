"""
Manufacturing OEE Analysis Demo

Demonstrates:
- Loading production data
- Detecting anomalies in production rate
- Calculating OEE (Overall Equipment Effectiveness) metrics
- Generating actionable recommendations

Usage:
    python examples/manufacturing_oee/run_demo.py

Good AI Philosophy: Leverage, not lore. Evidence over opinions.
"""

import json
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

import pandas as pd

from goodai.modules.anomaly_detection import AnomalyDetector


def calculate_oee(df: pd.DataFrame) -> dict:
    """
    Calculate Overall Equipment Effectiveness (OEE).

    OEE = Availability x Performance x Quality

    - Availability: Actual runtime / Planned runtime
    - Performance: Actual output / Theoretical output at full speed
    - Quality: Good units / Total units produced

    Args:
        df: DataFrame with columns:
            - uptime_minutes: actual production time
            - planned_minutes: scheduled production time
            - production_rate: actual units per hour
            - planned_rate: target units per hour
            - good_units: units passing quality check
            - total_units: total units produced

    Returns:
        Dictionary with availability, performance, quality, and overall OEE
    """
    # Availability = Uptime / Planned Time
    total_uptime = df["uptime_minutes"].sum()
    total_planned = df["planned_minutes"].sum()
    availability = total_uptime / total_planned if total_planned > 0 else 0

    # Performance = Actual Rate / Planned Rate (average across all periods)
    # Only count periods where there was uptime
    active_periods = df[df["uptime_minutes"] > 0]
    if len(active_periods) > 0:
        performance = (
            active_periods["production_rate"].sum() /
            active_periods["planned_rate"].sum()
        )
    else:
        performance = 0

    # Quality = Good Units / Total Units
    total_good = df["good_units"].sum()
    total_produced = df["total_units"].sum()
    quality = total_good / total_produced if total_produced > 0 else 0

    # Overall OEE
    overall = availability * performance * quality

    return {
        "availability": round(availability, 4),
        "performance": round(performance, 4),
        "quality": round(quality, 4),
        "overall": round(overall, 4),
    }


def generate_recommendations(oee: dict, anomalies: pd.DataFrame) -> list:
    """
    Generate actionable recommendations based on OEE analysis.

    Args:
        oee: OEE metrics dictionary
        anomalies: DataFrame with detected anomalies

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # OEE-based recommendations
    if oee["overall"] < 0.65:
        recommendations.append(
            "CRITICAL: OEE below 65%. Immediate action required. "
            "Focus on the lowest-scoring component first."
        )
    elif oee["overall"] < 0.75:
        recommendations.append(
            "WARNING: OEE below 75%. Review production processes for improvement opportunities."
        )
    elif oee["overall"] >= 0.85:
        recommendations.append(
            "OEE at world-class levels (85%+). Focus on maintaining consistency."
        )

    # Component-specific recommendations
    if oee["availability"] < 0.90:
        availability_pct = oee["availability"] * 100
        recommendations.append(
            f"Availability at {availability_pct:.1f}%. Investigate unplanned downtime causes. "
            "Consider preventive maintenance scheduling."
        )

    if oee["performance"] < 0.95:
        performance_pct = oee["performance"] * 100
        recommendations.append(
            f"Performance at {performance_pct:.1f}%. Look for speed losses and minor stoppages. "
            "Review operator training and equipment settings."
        )

    if oee["quality"] < 0.99:
        quality_pct = oee["quality"] * 100
        defect_rate = (1 - oee["quality"]) * 100
        recommendations.append(
            f"Quality at {quality_pct:.1f}% ({defect_rate:.2f}% defect rate). "
            "Implement statistical process control to reduce variation."
        )

    # Anomaly-based recommendations
    anomaly_count = anomalies["is_anomaly"].sum()
    if anomaly_count > 0:
        low_anomalies = anomalies[anomalies["anomaly_type"] == "low"]
        high_anomalies = anomalies[anomalies["anomaly_type"] == "high"]

        if len(low_anomalies) > 0:
            worst_drop = low_anomalies["value"].min()
            recommendations.append(
                f"Detected {len(low_anomalies)} production drops. "
                f"Lowest rate: {worst_drop:.0f} units/hour. "
                "Investigate root causes of these incidents."
            )

        if len(high_anomalies) > 0:
            recommendations.append(
                f"Detected {len(high_anomalies)} unusually high production periods. "
                "Verify data accuracy and investigate if sustainable."
            )

    if not recommendations:
        recommendations.append(
            "Production metrics within normal ranges. Continue monitoring."
        )

    return recommendations


def main():
    """Run the Manufacturing OEE Demo."""
    # Load configuration
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load sample data
    data_path = Path(__file__).parent / "sample_data.csv"
    df = pd.read_csv(data_path)

    # Initialize anomaly detector with config settings
    anomaly_config = config["anomaly_detection"]
    detector = AnomalyDetector(
        window_size=anomaly_config["window_size"],
        threshold=anomaly_config["threshold"]
    )

    # Detect anomalies in production rate
    anomalies = detector.detect(
        df,
        column=anomaly_config["target_column"],
        timestamp_column="timestamp"
    )

    # Calculate OEE metrics
    oee = calculate_oee(df)

    # Get anomaly details
    anomaly_rows = anomalies[anomalies["is_anomaly"]].head(5)
    anomaly_details = [
        {
            "timestamp": str(row["timestamp"]),
            "value": float(row["value"]),
            "type": row["anomaly_type"],
            "explanation": detector.explain_anomaly(row)
        }
        for _, row in anomaly_rows.iterrows()
    ]

    # Get anomaly summary
    anomaly_summary = detector.get_anomaly_summary(anomalies)

    # Generate recommendations
    recommendations = generate_recommendations(oee, anomalies)

    # Build result
    result = {
        "demo": config["name"],
        "data_points_analyzed": len(df),
        "oee_metrics": {
            "availability": f"{oee['availability'] * 100:.1f}%",
            "performance": f"{oee['performance'] * 100:.1f}%",
            "quality": f"{oee['quality'] * 100:.1f}%",
            "overall_oee": f"{oee['overall'] * 100:.1f}%"
        },
        "oee_raw": oee,
        "anomaly_detection": {
            "total_anomalies": anomaly_summary["anomalies_detected"],
            "anomaly_rate": f"{anomaly_summary['anomaly_rate'] * 100:.1f}%",
            "high_anomalies": anomaly_summary["high_anomalies"],
            "low_anomalies": anomaly_summary["low_anomalies"],
            "details": anomaly_details
        },
        "recommendations": recommendations,
        "methodology": "Good AI: Identify -> Pilot -> Instrument -> Learn -> Scale/Sunset"
    }

    # Output JSON
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    main()
