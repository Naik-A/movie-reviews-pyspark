# pyspark-movie-reviews

PySpark pipeline for the IMDb Dataset of 50K Movie Reviews (Data and Storytelling, Module 1, Homework 1.2).

The script loads the Kaggle CSV, prints the schema and a preview, drops empty reviews and labels, counts positive and negative reviews, adds a `review_length` column, shows reviews longer than 500 characters, and writes a cleaned CSV.

## Project layout

```text
pyspark-movie-reviews/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── .gitkeep
│   └── IMDB Dataset.csv    # you download this; it is not committed
├── output/                 # created when the script runs
└── src/
    └── process_reviews.py
```

## Setup (macOS)

PySpark 3.5 needs two things: **Java 17** and **Python 3.11**. Python 3.13 and newer are not supported. Check `python3 --version` first. If it prints 3.13 or higher, the commands below use `python3.11` on purpose.

```bash
brew install openjdk@17 python@3.11
export JAVA_HOME="$(brew --prefix openjdk@17)"
export PATH="$JAVA_HOME/bin:$PATH"
java -version
```

`brew --prefix` prints `/opt/homebrew` on Apple Silicon and `/usr/local` on Intel.

From this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download **IMDB Dataset.csv** from:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

Put it here, including the space in the file name:

```text
data/IMDB Dataset.csv
```

The CSV has two columns, `review` and `sentiment`. If the file is missing, the script stops before Spark starts and prints this same path.

## Run

From the repository root, with the virtual environment active:

```bash
python src/process_reviews.py
```

Optional arguments:

```bash
python src/process_reviews.py \
  --input "data/IMDB Dataset.csv" \
  --output "output/cleaned_imdb_reviews" \
  --min-length 500
```

The first run can take a minute while Spark starts. Later runs are faster.

Review text in the terminal is shortened so the run fits on one screen. The CSV still has every character.

## Expected output

The terminal is split into numbered sections:

1. **Load** — the CSV path and the Spark version.
2. **Explore** — `printSchema()` (string columns `review` and `sentiment`) and the first 5 rows, with each review shortened on screen.
3. **Clean** — row counts before cleaning, after cleaning, and how many rows were removed. Null, empty, and whitespace-only reviews or labels are removed. Sentiment is lowercased.
4. **Analyze** — total reviews (about 50,000 on the full Kaggle file), a sentiment breakdown, then separate positive and negative counts (about 25,000 each when the file is complete).
5. **Transform** — a `review_length` column (character length), the count of reviews longer than 500 characters, and a 5-row preview. The CSV keeps the full review.
6. **Save** — the output folder and the `part-*.csv` file inside it.
7. **SparkSession stopped.**

## Saved CSV

Spark writes a **directory** of files, not one `.csv` path:

```text
output/cleaned_imdb_reviews/
├── _SUCCESS
└── part-00000-<id>.csv
```

`coalesce(1)` forces a single part file so the result is easy to open. The part file includes a header and the columns `review`, `sentiment`, and `review_length`. Running the script again replaces that folder.

Peek at it with:

```bash
ls output/cleaned_imdb_reviews
head -n 3 output/cleaned_imdb_reviews/part-*.csv
```
