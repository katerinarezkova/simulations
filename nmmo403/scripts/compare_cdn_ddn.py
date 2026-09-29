from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Input
data_dir = Path("comparison")
cdn = pd.read_csv(data_dir / "CDN" / "diagnostics.csv")
ddn = pd.read_csv(data_dir / "DDN" / "diagnostics.csv")

# Output
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

plots = [
    ("kinetic_energy", "Kinetic energy", "kinetic_energy.png"),
    ("backflow_energy", "Backflow energy flux", "backflow_energy.png"),
    ("backflow", "Backflow volume flux", "backflow.png"),
]

last_cdn_time = cdn["time"].iloc[-1]

for column, ylabel, filename in plots:
    plt.figure(figsize=(8, 5))
    plt.plot(cdn["time"], cdn[column], label="CDN")
    plt.plot(ddn["time"], ddn[column], label="DDN")

    if last_cdn_time < ddn["time"].iloc[-1]:
        plt.axvline(
            last_cdn_time,
            color="gray",
            linestyle="--",
            label=f"Last successful CDN step (t={last_cdn_time:.2f})",
        )

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"CDN vs DDN: {ylabel}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()

print(f"Saved to {output_dir.resolve()}")
