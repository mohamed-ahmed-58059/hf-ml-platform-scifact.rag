"""Pretty-prints per-experiment rows and the final master table."""
from tabulate import tabulate


class ResultsTablePrinter:
    """Per-experiment immediate row + final cross-experiment master table."""

    def print_row(self, name: str, metrics: dict[str, float]) -> None:
        keys = sorted(metrics)
        row = [name] + [f"{metrics[k]:.4f}" for k in keys]
        print("\n" + tabulate([row], headers=["Experiment"] + keys, tablefmt="grid"))

    def print_master(self, results: list[tuple[str, dict[str, float]]]) -> None:
        if not results:
            print("[Master] (no results)")
            return
        all_keys = sorted({k for _, m in results for k in m})
        rows = []
        for name, metrics in results:
            row = [name]
            for k in all_keys:
                v = metrics.get(k)
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
            rows.append(row)
        print("\n=== Master Table ===")
        print(tabulate(rows, headers=["Experiment"] + all_keys, tablefmt="grid"))
