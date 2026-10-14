"""Build the cleaned dataset and the three report plots.

Usage: python -m src.air_travel [--data-dir DIR] [--output-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.air_travel import analysis, data, plots  # noqa: E402
from src.paths import ARTIFACTS_DIR, ensure_dir  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="air_travel", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "air_travel")
    args = parser.parse_args(argv)
    out = ensure_dir(args.output_dir)

    holidays, passengers = data.load_raw(args.data_dir)
    df = data.build_dataset(holidays, passengers)
    df.to_csv(out / "clean_dataset.csv", index=False)
    print(f"Clean dataset: {len(df)} country-months")
    print(df.describe())

    growth = analysis.regional_growth(df)
    monthly = analysis.monthly_growth_index(df)

    for name, fig in [
        ("plot1_growth_trends.png", plots.plot_growth_trends(growth)),
        ("plot2_holiday_relationship.png", plots.plot_holiday_relationship(df)),
    ]:
        fig.savefig(out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    plots.plot_growth_vs_holidays(monthly).write_html(out / "plot3_growth_vs_holidays.html")

    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
