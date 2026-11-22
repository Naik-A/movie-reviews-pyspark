#!/usr/bin/env python3
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT /"data"/ "IMDB Dataset.csv"
OUTPUT_DIR = ROOT / "output" / "cleaned_imdb_reviews"
PREVIEW_CHARS = 48
def main() -> None:
    if not INPUT_CSV.is_file():
        raise SystemExit("Not Found")
    spark = (
        SparkSession.builder.appName("IMDb Movie Reviews")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config(
            "spark.driver.extraJavaOptions",
            f"-Dlog4j2.configurationFile={(ROOT / 'log4j2.properties').resolve()}",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark version: {spark.version}")

    try:
        print("\n=== 1. Load the dataset ===")
        print("Reading: data/IMDB Dataset.csv")
        df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("multiLine", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(INPUT_CSV))
        )
        df = df.select("review", "sentiment")

        print("\n=== 2. Explore the data ===")
        df.printSchema()
        print(f"First 5 rows (first {PREVIEW_CHARS} characters of each review):")
        df.select(
            "sentiment",
            F.substring("review", 1, PREVIEW_CHARS).alias("review"),
        ).show(5, truncate=False)

        print("\n=== 3. Clean the data ===")
        before = df.count()
        df = (
            df.withColumn("review", F.trim("review"))
            .withColumn("sentiment", F.lower(F.trim("sentiment")))
            .filter(F.length("review") > 0)
            .filter(F.length("sentiment") > 0)
        )
        after = df.count()
        print(f"Rows before cleaning: {before:,}")
        print(f"Rows after cleaning:  {after:,}")
        print(f"Rows removed:         {before - after:,}")

        print("\n=== 4. Analyze the data ===")
        print(f"Total reviews: {after:,}")
        counts = {
            row["sentiment"]: row["count"]
            for row in df.groupBy("sentiment").count().collect()
        }
        print(f"Positive reviews: {counts.get('positive', 0):,}")
        print(f"Negative reviews: {counts.get('negative', 0):,}")

        print("\n=== 5. Transform the data ===")
        print("Added column: review_length (number of characters)")
        df = df.withColumn("review_length", F.length("review"))
        long_reviews = df.filter(F.col("review_length") > 500)
        print(f"Reviews with review_length > 500: {long_reviews.count():,}")
        print("Five longest matching reviews:")
        long_reviews.orderBy(F.desc("review_length")).select(
            "sentiment",
            "review_length",
            F.substring("review", 1, PREVIEW_CHARS).alias("review"),
        ).show(5, truncate=False)

        print("\n=== 6. Save the results ===")
        print("Writing: output/cleaned_imdb_reviews/")
        OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
        (
            df.coalesce(1)
            .write.mode("overwrite")
            .option("header", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(str(OUTPUT_DIR))
        )
        print("Saved 1 CSV file in output/cleaned_imdb_reviews/")
    finally:
        spark.stop()
        print("\nSparkSession stopped.")
    print("Done.")


if __name__ == "__main__":
    main()
