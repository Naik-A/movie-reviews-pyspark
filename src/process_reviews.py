#!/usr/bin/env python3
"""Clean and summarize the IMDb 50K movie-review dataset with PySpark.

Pipeline
--------
1. Start a local SparkSession.
2. Load ``data/IMDB Dataset.csv`` (header, inferred types, multiline fields).
3. Print the schema and a short preview.
4. Drop rows with a missing review or sentiment label.
5. Count all reviews, then positive and negative labels.
6. Add ``review_length`` and show reviews longer than 500 characters.
7. Write the result to ``output/cleaned_imdb_reviews/`` and stop Spark.

Run from the repository root::

    python src/process_reviews.py
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# PySpark 3.5 supports Python 3.9–3.12. Newer macOS installs often default to
# Python 3.13+, which fails later with a much harder-to-read error.
if sys.version_info < (3, 9) or sys.version_info >= (3, 13):
    current = ".".join(str(part) for part in sys.version_info[:3])
    sys.exit(
        "This script uses PySpark 3.5, which supports Python 3.9 through 3.12.\n"
        f"The current interpreter is Python {current}.\n\n"
        "Create a virtual environment with Python 3.11, then run the script again:\n"
        "  brew install python@3.11\n"
        "  python3.11 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "  python src/process_reviews.py"
    )

try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
except ImportError:
    sys.exit(
        "PySpark is not installed for this Python.\n\n"
        "From the repository root, with Python 3.11:\n"
        "  python3.11 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        "  python src/process_reviews.py"
    )

# Repository root is the parent of src/, so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "IMDB Dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "cleaned_imdb_reviews"
DEFAULT_MIN_LENGTH = 500
# Characters of each review shown in the terminal. The CSV keeps the full text.
PREVIEW_CHARS = 48

# Printed when the CSV has not been downloaded yet.
DATASET_HELP = """
Could not find the IMDb CSV.

Download "IMDB Dataset.csv" from Kaggle:
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Place the file here (the space in the name is required):
  data/IMDB Dataset.csv

Or pass a different path:
  python src/process_reviews.py --input "/path/to/IMDB Dataset.csv"
""".strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read optional input, output, and length-filter settings."""
    parser = argparse.ArgumentParser(
        description="Clean and summarize the IMDb 50K movie reviews with PySpark."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to IMDB Dataset.csv (default: data/IMDB Dataset.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directory where Spark writes the cleaned CSV (default: output/cleaned_imdb_reviews).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help="Show reviews with more characters than this value (default: 500).",
    )
    return parser.parse_args(argv)


def section(title: str) -> None:
    """Print a visible heading so each homework step is easy to find."""
    print(f"\n=== {title} ===")


def resolve_input_path(raw_path: str) -> Path:
    """Return the CSV path, or raise FileNotFoundError with download steps."""
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{DATASET_HELP}\n\nLooked for:\n  {path}")
    return path


def create_spark_session() -> SparkSession:
    """Start a local Spark session and keep its logs off the screen.

    ``local[*]`` uses every CPU core on this machine. The shuffle setting is
    lowered because this file has only about 50,000 rows.
    """
    log_config = (PROJECT_ROOT / "src" / "log4j2.properties").resolve()
    spark = (
        SparkSession.builder.appName("IMDb Movie Reviews")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config(
            "spark.driver.extraJavaOptions",
            f"-Dlog4j2.configurationFile={log_config}",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark version: {spark.version}")
    return spark


def load_reviews(spark: SparkSession, csv_path: Path) -> DataFrame:
    """Load the Kaggle CSV into a DataFrame.

    Reviews are quoted and can contain commas, quotation marks, and newlines,
    so multiline parsing and quote/escape options stay on.
    """
    section("1. Load the dataset")
    print(f"Reading: {display_path(csv_path)}")

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        # Keep a bad row as nulls instead of crashing the whole load.
        .option("mode", "PERMISSIVE")
        .csv(str(csv_path))
    )
    return standardize_columns(df)


def standardize_columns(df: DataFrame) -> DataFrame:
    """Keep ``review`` and ``sentiment``, even if the header has a UTF-8 BOM.

    Some spreadsheet exports name the first column ``\\ufeffreview``. Renaming
    that back to ``review`` lets the rest of the script stay simple.
    """
    cleaned_names = [column.lstrip("\ufeff").strip() for column in df.columns]
    df = df.toDF(*cleaned_names)

    lookup = {column.lower(): column for column in df.columns}
    for expected in ("review", "sentiment"):
        actual = lookup.get(expected)
        if actual is not None and actual != expected:
            df = df.withColumnRenamed(actual, expected)

    missing = [name for name in ("review", "sentiment") if name not in df.columns]
    if missing:
        found = ", ".join(df.columns) or "(none)"
        raise ValueError(
            "This CSV does not look like the IMDb review file. "
            f"Missing column(s): {', '.join(missing)}. Found: {found}."
        )

    # Ignore any extra columns so later steps have a stable schema.
    return df.select("review", "sentiment")


def display_path(path: Path) -> str:
    """Prefer a short project-relative path in the terminal."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def preview(df: DataFrame, *columns: str) -> DataFrame:
    """Keep the requested columns and a short slice of the review text."""
    selected = [F.col(name) for name in columns]
    selected.append(F.substring(F.col("review"), 1, PREVIEW_CHARS).alias("review"))
    return df.select(*selected)


def explore(df: DataFrame) -> None:
    """Print the schema and the first five reviews, shortened to fit a screen."""
    section("2. Explore the data")
    df.printSchema()
    print(f"First 5 rows (first {PREVIEW_CHARS} characters of each review):")
    preview(df, "sentiment").show(5, truncate=False)


def clean(df: DataFrame) -> DataFrame:
    """Drop rows whose review or sentiment is null or only whitespace.

    Sentiment is stored in lowercase so later counts treat ``Positive`` and
    ``positive`` as the same label.
    """
    section("3. Clean the data")
    before = df.count()

    review = F.trim(F.col("review"))
    sentiment = F.lower(F.trim(F.col("sentiment")))

    cleaned = (
        df.withColumn("review", review)
        .withColumn("sentiment", sentiment)
        # length() > 0 is false for null and for strings that became empty
        # after trim, so one filter covers null, "", and whitespace-only text.
        .filter(F.length(F.col("review")) > 0)
        .filter(F.length(F.col("sentiment")) > 0)
    )

    after = cleaned.count()
    print(f"Rows before cleaning: {before:,}")
    print(f"Rows after cleaning:  {after:,}")
    print(f"Rows removed:         {before - after:,}")
    return cleaned


def analyze(df: DataFrame) -> None:
    """Count every review, then the positive and negative labels."""
    section("4. Analyze the data")

    total = df.count()
    print(f"Total reviews: {total:,}")

    # One grouped count covers both labels and any unexpected sentiment values.
    label_counts = df.groupBy("sentiment").count().orderBy(F.desc("count"))
    counts = {row["sentiment"]: row["count"] for row in label_counts.collect()}
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    other = total - positive - negative

    print(f"Positive reviews: {positive:,}")
    print(f"Negative reviews: {negative:,}")
    if other:
        print(f"Other labels:     {other:,}")


def add_review_length(df: DataFrame) -> DataFrame:
    """Add ``review_length``, the number of characters in each review."""
    return df.withColumn("review_length", F.length(F.col("review")))


def show_long_reviews(df: DataFrame, min_length: int) -> None:
    """Print how many reviews exceed ``min_length`` and preview five of them."""
    section("5. Transform the data")
    print("Added column: review_length (number of characters)")

    long_reviews = df.filter(F.col("review_length") > min_length)
    long_count = long_reviews.count()
    print(f"Reviews with review_length > {min_length}: {long_count:,}")

    if long_count == 0:
        print("No reviews matched that length filter.")
        return

    # The saved CSV keeps the full review. The terminal only needs a preview.
    print("Five longest matching reviews:")
    (
        preview(long_reviews.orderBy(F.desc("review_length")), "sentiment", "review_length")
        .show(5, truncate=False)
    )


def save_results(df: DataFrame, output_dir: Path) -> None:
    """Write one CSV part file inside ``output_dir``.

    Spark always writes a folder, not a single file. ``coalesce(1)`` puts every
    row into one partition, so the folder contains one ``part-*.csv`` file plus
    a ``_SUCCESS`` marker. That part file has a header and can be opened in
    Excel or pandas. Overwrite lets you run the script again safely.
    """
    section("6. Save the results")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing: {display_path(output_dir)}/")

    (
        df.select("review", "sentiment", "review_length")
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        # Double any quote inside a review (" -> "") so Excel and pandas
        # open the file as ordinary CSV. Spark's default escape is a backslash.
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(output_dir))
    )

    part_files = sorted(output_dir.glob("part-*.csv"))
    if not part_files:
        raise RuntimeError(f"Spark finished, but no CSV was written in {output_dir}")

    print(f"Saved {len(part_files)} CSV file in {display_path(output_dir)}/")


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline and always stop Spark if it was started."""
    args = parse_args(argv)

    try:
        input_path = resolve_input_path(args.input)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if args.min_length < 0:
        print("--min-length must be 0 or greater.")
        return 1

    spark: SparkSession | None = None
    try:
        spark = create_spark_session()
        reviews = load_reviews(spark, input_path)
        explore(reviews)

        # Cache the cleaned rows. The counts and the length filter both read them.
        cleaned = clean(reviews)
        cleaned.cache()
        analyze(cleaned)

        transformed = add_review_length(cleaned)
        show_long_reviews(transformed, args.min_length)
        save_results(transformed, Path(args.output).expanduser().resolve())
        cleaned.unpersist()
    except (ValueError, RuntimeError) as exc:
        print(f"\n{exc}")
        return 1
    except Exception:
        print(
            "\nThe Spark job failed before it could finish.\n"
            "If the traceback mentions Java, install JDK 17 and follow README.md.\n"
        )
        traceback.print_exc()
        return 1
    finally:
        if spark is not None:
            spark.stop()
            print("\nSparkSession stopped.")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
