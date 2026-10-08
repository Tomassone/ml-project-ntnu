"""Build a leakage-conscious long table for SINTEF Tokke-Vinje unit commitment.

One row = (Run No, generator, hour 0..167), one binary target.
This module intentionally contains NO model training or feature selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

HOURS = 168
CSV_STEMS = {
    "decisions": "Unit_commitment_decisions",
    "price": "Historical_day_ahead_price_2015_2025",
    "inflow": "Historical_inflow_1958_2025",
    "volume": "Historical_volume_2015_2024",
    "water_value": "Synthetic_water_value_2015_2024",
    "min_flow": "Constraint_min_flow",
    "min_volume": "Constraint_min_volume",
}


def find_input(data_dir: Path, stem: str, suffix: str) -> Path:
    """Allow official filenames or the '(1)' suffix added by downloads."""
    direct = data_dir / (stem + suffix)
    if direct.exists():
        return direct
    matches = sorted(data_dir.glob(stem + "(*)" + suffix))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Expected {direct} or exactly one {stem}(N){suffix} in {data_dir}. "
        f"Found: {[p.name for p in matches]}"
    )


def _read_time_table(path: Path, min_time: pd.Timestamp, max_time: pd.Timestamp,
                     columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=columns)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.loc[df["date"].between(min_time, max_time)].set_index("date")
    if not df.index.is_unique:
        raise ValueError(f"Duplicate timestamps in {path}")
    return df.astype("float32")


def _read_inflow(path: Path, min_time: pd.Timestamp, max_time: pd.Timestamp) -> pd.DataFrame:
    """Read only the required period instead of retaining all hourly rows since 1958."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = {c: "float32" for c in columns if c != "date"}
    parts = []
    for chunk in pd.read_csv(path, chunksize=100_000, dtype=dtypes):
        # The ISO 8601 date strings sort chronologically; parse only retained rows.
        keep = chunk["date"].str[:10].between(
            min_time.strftime("%Y-%m-%d"), max_time.strftime("%Y-%m-%d")
        )
        subset = chunk.loc[keep].copy()
        if not subset.empty:
            subset["date"] = pd.to_datetime(subset["date"], utc=True)
            subset = subset.loc[subset["date"].between(min_time, max_time)]
            parts.append(subset)
    if not parts:
        raise ValueError("No inflow data found for selected cases")
    inflow = pd.concat(parts, ignore_index=True).set_index("date")
    if not inflow.index.is_unique:
        raise ValueError("Duplicate timestamps in inflow")
    return inflow


def load_data(data_dir: Path, limit_cases: int | None = None) -> dict:
    """Load the original input tables and restrict time series to the actual cases."""
    data_dir = Path(data_dir)
    paths = {key: find_input(data_dir, stem, ".csv") for key, stem in CSV_STEMS.items()}
    topo_path = find_input(data_dir, "Tokke_Vinje_topology", ".yaml")

    all_columns = pd.read_csv(paths["decisions"], nrows=0).columns.tolist()
    target_cols = [c for c in all_columns if c.startswith("result_committed_")]
    decision_types = {c: "uint8" for c in target_cols}
    decision_types.update({"Run No": "int32", "total_calculation_time": "float32"})
    decisions = pd.read_csv(paths["decisions"], nrows=limit_cases, dtype=decision_types)
    decisions["starttime"] = pd.to_datetime(decisions["starttime"], utc=True)
    if not decisions["Run No"].is_unique:
        raise ValueError("Run No is not unique")

    with topo_path.open(encoding="utf-8") as file:
        topology = yaml.safe_load(file)

    generators = list(topology["model"]["generator"].keys())
    target_lookup = {
        gen: [f"result_committed_{gen}_t{h}" for h in range(HOURS)]
        for gen in generators
    }
    target_expected = {name for cols in target_lookup.values() for name in cols}
    if set(target_cols) != target_expected:
        missing = target_expected - set(target_cols)
        extra = set(target_cols) - target_expected
        raise ValueError(f"Target columns do not match YAML generators: missing={missing}, extra={extra}")
    if decisions[target_cols].isna().any().any():
        raise ValueError("Missing target values")
    if not decisions[target_cols].isin([0, 1]).all().all():
        raise ValueError("Targets must be 0 or 1")

    min_start = decisions["starttime"].min()
    max_hour = decisions["starttime"].max() + pd.Timedelta(hours=HOURS-1)
    max_end = decisions["starttime"].max() + pd.Timedelta(hours=HOURS)

    price = _read_time_table(paths["price"], min_start, max_hour)
    price = price.rename(columns={"Day-ahead price (EUR/MWh)": "price_current"})
    inflow = _read_inflow(paths["inflow"], min_start, max_hour)
    min_flow = _read_time_table(paths["min_flow"], min_start, max_hour)
    min_volume = _read_time_table(paths["min_volume"], min_start, max_hour)
    volume = _read_time_table(paths["volume"], min_start, decisions["starttime"].max())
    water_value = _read_time_table(paths["water_value"], min_start, max_end)

    if "price_current" not in price:
        raise ValueError("Price column missing")
    res_names = set(topology["model"]["reservoir"])
    for key, df in [("inflow", inflow), ("volume", volume), ("water_value", water_value)]:
        if not res_names.issubset(df.columns):
            raise ValueError(f"Missing reservoir columns in {key}: {sorted(res_names-set(df.columns))}")

    return dict(decisions=decisions, target_cols=target_cols, target_lookup=target_lookup,
                generators=generators, topology=topology, price=price, inflow=inflow,
                min_flow=min_flow, min_volume=min_volume, volume=volume, water_value=water_value)


def topology_metadata(topology: dict) -> pd.DataFrame:
    """Direct reservoirs and ALL upstream reservoirs (direct included).

    Follows directed YAML edges, including tunnels and rivers. Reservoirs reached via
    different paths are counted only once. In particular Songa has two direct reservoirs.
    """
    predecessors = defaultdict(list)
    for edge in topology["connections"]:
        a = (edge["from_type"], edge["from"])
        b = (edge["to_type"], edge["to"])
        predecessors[b].append(a)

    plant_of_gen = {
        edge["from"]: edge["to"]
        for edge in topology["connections"]
        if edge["from_type"] == "generator" and edge["to_type"] == "plant"
    }
    capacities = {res: value["max_vol"] for res, value in topology["model"]["reservoir"].items()}
    records = []
    for gen, spec in topology["model"]["generator"].items():
        plant = plant_of_gen[gen]
        direct = set()
        for node_type, node_name in predecessors[("plant", plant)]:
            if node_type == "reservoir":
                direct.add(node_name)
            if node_type == "tunnel":
                direct.update(name for kind, name in predecessors[("tunnel", node_name)]
                              if kind == "reservoir")

        visited = set()
        stack = [("plant", plant)]
        upstream = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node[0] == "reservoir":
                upstream.add(node[1])
            stack.extend(predecessors[node])

        if not direct or not direct.issubset(upstream):
            raise ValueError(f"Bad reservoir mapping for {gen}: {direct}, {upstream}")
        records.append({
            "generator": gen, "plant": plant,
            "direct_reservoirs": sorted(direct),
            "upstream_reservoirs": sorted(upstream),
            "direct_capacity": sum(capacities[r] for r in direct),
            "p_min": float(spec["p_min"]), "p_max": float(spec["p_max"]),
            "p_nom": float(spec["p_nom"]),
            "start_cost": float(spec["startcost_const"]),
            "stop_cost": float(spec["stopcost_const"]),
            "penstock": int(spec["penstock"]),
        })
    return pd.DataFrame(records).set_index("generator", drop=False)


def _weighted(values: pd.DataFrame, volumes: pd.DataFrame, selected: list[str]) -> np.ndarray:
    """Marginal water value weighted by initial reservoir volumes (zero total -> NaN)."""
    v = volumes[selected].to_numpy(dtype="float64")
    x = values[selected].to_numpy(dtype="float64")
    num = (v * x).sum(axis=1)
    denom = v.sum(axis=1)
    return np.divide(num, denom, out=np.full(len(denom), np.nan), where=denom > 0).astype("float32")


def make_batch(data: dict, metadata: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Produce a batch without materializing the full multi-million-row table in RAM."""
    n = len(cases)
    hours = np.tile(np.arange(HOURS, dtype="uint8"), n)
    starts = pd.DatetimeIndex(np.repeat(cases["starttime"].to_numpy(), HOURS))
    times = starts + pd.to_timedelta(hours.astype("int16"), unit="h")
    t0 = pd.DatetimeIndex(cases["starttime"])
    t_end = t0 + pd.Timedelta(hours=HOURS)

    hourly_price = data["price"].reindex(times)["price_current"]
    hourly_inflow = data["inflow"].reindex(times)
    hourly_min_flow = data["min_flow"].reindex(times)
    hourly_min_volume = data["min_volume"].reindex(times)
    initial_volumes = data["volume"].reindex(t0)
    final_water_value = data["water_value"].reindex(t_end)
    sources = [hourly_price, hourly_inflow, hourly_min_flow,
               hourly_min_volume, initial_volumes, final_water_value]
    if any(source.isna().any().any() for source in sources):
        raise ValueError("Time alignment produced missing values; inspect input dates before proceeding")

    common = {
        "run_no": np.repeat(cases["Run No"].to_numpy(dtype="int32"), HOURS),
        "starttime": starts,
        "hour": hours,
        "timestamp": times,
        "price_current": hourly_price.to_numpy(dtype="float32"),
    }
    for column in hourly_min_volume.columns:
        common[f"min_volume_{column}"] = hourly_min_volume[column].to_numpy(dtype="float32")
    for column in hourly_min_flow.columns:
        common[f"min_flow_{column}"] = hourly_min_flow[column].to_numpy(dtype="float32")
    # Only river with exogenous inflow: keep as a separate global input.
    common["inflow_r_Vest_Vassdraget"] = hourly_inflow["r_Vest_Vassdraget"].to_numpy(dtype="float32")

    frames = []
    for gen in data["generators"]:
        m = metadata.loc[gen]
        direct = m["direct_reservoirs"]
        upstream = m["upstream_reservoirs"]  # upstream INCLUDES direct reservoirs

        direct_volume = initial_volumes[direct].sum(axis=1).to_numpy(dtype="float32")
        upstream_volume = initial_volumes[upstream].sum(axis=1).to_numpy(dtype="float32")
        direct_water = _weighted(final_water_value, initial_volumes, direct)
        upstream_water = _weighted(final_water_value, initial_volumes, upstream)
        direct_inflow = hourly_inflow[direct].sum(axis=1).to_numpy(dtype="float32")
        upstream_inflow = hourly_inflow[upstream].sum(axis=1).to_numpy(dtype="float32")

        gen_cols = {
            "generator": gen, "plant": m["plant"],
            "p_min": np.float32(m["p_min"]), "p_max": np.float32(m["p_max"]),
            "p_nom": np.float32(m["p_nom"]),
            "start_cost": np.float32(m["start_cost"]),
            "stop_cost": np.float32(m["stop_cost"]),
            "penstock": np.uint8(m["penstock"]),
            "direct_total_volume": np.repeat(direct_volume, HOURS),
            # Capacity-weighted mean fill ratio, not an unweighted mean of ratios.
            "direct_mean_fill_ratio": np.repeat(direct_volume / m["direct_capacity"], HOURS).astype("float32"),
            "direct_total_inflow": direct_inflow,
            "direct_mean_water_value": np.repeat(direct_water, HOURS),
            "upstream_total_volume": np.repeat(upstream_volume, HOURS),
            "upstream_total_inflow": upstream_inflow,
            "upstream_mean_water_value": np.repeat(upstream_water, HOURS),
            "target": cases[data["target_lookup"][gen]].to_numpy(dtype="uint8").reshape(-1),
        }
        frames.append(pd.DataFrame({**common, **gen_cols}))

    batch = pd.concat(frames, ignore_index=True)
    # Ordering isn't necessary for ML, but makes the file easy to inspect and audit.
    gen_order = pd.Categorical(batch["generator"], categories=data["generators"], ordered=True)
    batch = batch.assign(_gen_order=gen_order).sort_values(
        ["run_no", "_gen_order", "hour"], kind="stable"
    ).drop(columns="_gen_order").reset_index(drop=True)
    batch["generator"] = pd.Categorical(batch["generator"], categories=data["generators"])
    batch["plant"] = batch["plant"].astype("category")
    if batch.duplicated(["run_no", "generator", "hour"]).any():
        raise ValueError("Long keys are not unique")
    if len(batch) != n * len(data["generators"]) * HOURS:
        raise ValueError("Unexpected number of rows")
    return batch


def to_wide(long_table: pd.DataFrame, target_columns: list[str],
            value_column: str = "target", metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Invert long predictions back into original 2352-column format (no training here).

    For a submission use value_column='prediction', after attaching predictions.
    total_calculation_time is never a feature; supply metadata only when comparing
    back to the training CSV. It will be NaN for new/test cases without this metadata.
    """
    key = ["run_no", "starttime", "generator", "hour"]
    small = long_table[key + [value_column]].copy()
    if small.duplicated(key).any():
        raise ValueError("Duplicated prediction keys")
    small["target_column"] = (
        "result_committed_" + small["generator"].astype(str)
        + "_t" + small["hour"].astype(str)
    )
    wide = small.pivot(index=["run_no", "starttime"],
                       columns="target_column", values=value_column)
    if not set(target_columns).issubset(wide.columns):
        raise ValueError("Missing predictions for at least one generator-hour column")
    wide = wide.reindex(columns=target_columns).reset_index()
    wide = wide.rename(columns={"run_no": "Run No", "starttime": "starttime"})
    if metadata is not None:
        calculation = metadata[["Run No", "total_calculation_time"]].copy()
        wide = wide.merge(calculation, on="Run No", validate="one_to_one")
    else:
        wide["total_calculation_time"] = np.nan
    return wide[["Run No", "starttime", "total_calculation_time"] + target_columns]


def build_dataset(data_dir: Path, output_file: Path, *, limit_cases: int | None = None,
                  batch_cases: int = 32, output_format: str = "parquet") -> dict:
    """Write the data incrementally. Parquet recommended; CSV fallback for environments without pyarrow."""
    data = load_data(data_dir, limit_cases=limit_cases)
    mapping = topology_metadata(data["topology"])
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if batch_cases < 1:
        raise ValueError("batch_cases must be positive")

    writer = None
    if output_format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Parquet requires 'pip install pyarrow'. Alternatively use --format csv") from exc
    elif output_format != "csv":
        raise ValueError("output_format must be 'parquet' or 'csv'")

    rows = 0
    sample = None
    try:
        for start in range(0, len(data["decisions"]), batch_cases):
            cases = data["decisions"].iloc[start:start + batch_cases]
            chunk = make_batch(data, mapping, cases)
            if sample is None:
                sample = chunk.head(5)
                # Exact round trip check on one batch, including original column order.
                restored = to_wide(chunk, data["target_cols"], metadata=cases)
                expected = cases[["Run No", "starttime", "total_calculation_time"] + data["target_cols"]]
                pd.testing.assert_frame_equal(restored.reset_index(drop=True),
                                              expected.reset_index(drop=True), check_dtype=False,
                                              check_names=False)
            if output_format == "parquet":
                tbl = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, tbl.schema, compression="zstd")
                writer.write_table(tbl)
            else:
                chunk.to_csv(output_file, mode="w" if start == 0 else "a",
                             header=(start == 0), index=False)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    return {"rows": rows, "cases": len(data["decisions"]),
            "generators": len(data["generators"]), "features": sample.columns.tolist(),
            "sample": sample, "generator_mapping": mapping,
            "output": output_file}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True, help="Folder with 7 CSVs and topology YAML")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit-cases", type=int, default=None, help="Try 2 first, then omit for all cases")
    p.add_argument("--batch-cases", type=int, default=32)
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = p.parse_args()
    result = build_dataset(args.data_dir, args.out, limit_cases=args.limit_cases,
                           batch_cases=args.batch_cases, output_format=args.format)
    print(f"Created: {result['output']} | {result['rows']:,} rows | {len(result['features'])} cols")
    print(result["sample"].to_string(index=False))


if __name__ == "__main__":
    main()
