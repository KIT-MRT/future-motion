import wandb
import pandas as pd


def format_dict_to_table(metrics_dict):
    """
    Formats the Waymo prediction metrics into a table format.
    :param data: Dictionary containing Waymo metrics.
    :return: Formatted DataFrame.
    """
    waymo_prefix = ["waymo_pred", "waymo_metrics"]
    agents = ["vehicle", "pedestrian", "cyclist", "ego"]
    metrics = [
        "mean_average_precision",
        "min_ade",
        "min_fde",
        "miss_rate",
        "overlap_rate",
    ]
    time = {"3": "5", "5": "9", "8": "15"}

    rows = []

    # Parse average values
    for agent in agents:
        ego = "ego_" if agent == "ego" else ""
        agent_abbrev = agent.upper() if agent != "ego" else "VEHICLE"
        for s, t in time.items():
            row = {"Object Type": agent.capitalize(), "Measurement time (s)": s}
            for metric in metrics:
                metric_name = f"waymo_{ego}pred_{metric}_TYPE_{agent_abbrev}_{t}"
                value = metrics_dict.get(f"waymo_metrics/{metric_name}")
                if value is not None:
                    row[metric] = round(value, 4)
            rows.append(row)

        agent_abbrev = agent[:3] if agent != "ego" else "veh"
        row = {"Object Type": agent.capitalize(), "Measurement time (s)": "Avg"}
        for metric in metrics:
            metric_name = f"{agent_abbrev}/{metric}"
            value = metrics_dict.get(f"waymo_{ego}pred/{metric_name}")
            if value is not None:
                row[metric] = round(value, 4)
        rows.append(row)

    # Create DataFrame
    table = pd.DataFrame(rows)

    # Add "All" class (avg. over all classes)
    for s, t in time.items():
        row = {"Object Type": "All", "Measurement time (s)": s}
        for metric in metrics:
            metric_prefix = f"waymo_metrics/waymo_pred_{metric}"
            value = (
                metrics_dict.get(f"{metric_prefix}_TYPE_VEHICLE_{t}")
                + metrics_dict.get(f"{metric_prefix}_TYPE_PEDESTRIAN_{t}")
                + metrics_dict.get(f"{metric_prefix}_TYPE_CYCLIST_{t}")
            ) / 3
            if value is not None:
                row[metric] = round(value, 4)
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)

    row = {"Object Type": "All", "Measurement time (s)": "Avg"}
    for metric in metrics:
        value = (
            metrics_dict.get(f"waymo_pred/{metric}")
            + metrics_dict.get(f"waymo_pred/{metric}")
            + metrics_dict.get(f"waymo_pred/{metric}")
        ) / 3
        if value is not None:
            row[metric] = round(value, 4)
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    return table


def export_metrics(metrics, output_file):
    with open(output_file, "w") as f:
        f.write(metrics.to_string())


if __name__ == "__main__":
    run_id = "3lk158lm"
    output_file = (
        f"./future-motion/future_motion/metrics/waymo_submission_metrics_{run_id}.txt"
    )

    # Login to WandB
    wandb.login()
    # Initialize API
    api = wandb.Api()
    # Fetch the specific run (replace `entity`, `project`, and `run_id` with your details)
    run = api.run(f"entity/project/{run_id}")

    # Access the metrics logged during training
    history_dict = run.summary

    # Filter the dictionary to only include keys that start with 'waymo_pred' or 'waymo_metrics'
    filtered_history_dict = {
        k: v
        for k, v in history_dict.items()
        if k.startswith("waymo_pred") or k.startswith("waymo_metrics")
    }

    waymo_submission_table = format_dict_to_table(history_dict)

    export_metrics(waymo_submission_table, output_file)
