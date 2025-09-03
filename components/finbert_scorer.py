from datasets import load_dataset
from pathlib import Path
from torch.amp import autocast
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, pipeline)

from typing import List, Optional
import math
import polars as pl
import torch

from components.article_preprocessor import ArticlePreprocessor
from components.schema import Schema
from components.settings import Settings
from utils.logger import Logger

class NSIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class FinBertScorer:
    """
    ProsusAI/finbert off-the-shelf; returns p_pos, p_neg, p_neu, sentiment_score=p_pos-p_neg
    """
    def __init__(self, batch_size_: int, max_length_: int, model_=None, tokenizer_=None):
        self.device = torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps")
            if (torch.backends.mps.is_available() and torch.backends.mps.is_built())
            else torch.device("cpu"))

        Logger.info(f"Using device {self.device}")

        if model_ is None:
            model_id = "ProsusAI/finbert"
            mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
            tok = AutoTokenizer.from_pretrained(model_id)
        else:
            mdl = model_
            # If caller didn’t pass a tokenizer, fall back to matching one
            tok = tokenizer_ or AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.pipe = pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            device=self.device,
            top_k=None,
            truncation=True
        )
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            self.pipe.model = torch.compile(self.pipe.model) # Compile the model
        self.batch_size = batch_size_
        # clamp to tokenizer's configured limit
        try:
            self.max_length = min(max_length_, getattr(self.pipe.tokenizer, "model_max_length", max_length_) or max_length_)
        except Exception:
            self.max_length = max_length_

    def score_texts(self, texts_: List[str]) -> pl.DataFrame:
        if not texts_:
            return pl.DataFrame(schema=["p_pos", "p_neg", "p_neu", "sentiment_score"])
        results = []
        self.pipe.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts_), self.batch_size), desc="FinBERT", unit="batch"):
                batch = texts_[i:i + self.batch_size]
                inputs = self.pipe.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.pipe.device) for k, v in inputs.items()}
                if self.device.type == "cuda":
                    with autocast(device_type="cuda"): # Enable mixed precision
                        outputs = self.pipe.model(**inputs)
                else:
                    with torch.no_grad():
                        outputs = self.pipe.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                for prob in probs:
                    p_pos, p_neg, p_neu = prob
                    results.append([p_pos, p_neg, p_neu, p_pos - p_neg])
        return pl.DataFrame(results, schema=["p_pos", "p_neg", "p_neu", "sentiment_score"], orient="row")

    @staticmethod
    def compute_nsi(prices_df_: pl.DataFrame, schema_: Schema, threshold_: float = 0.01) -> pl.DataFrame:
        prices = prices_df_
        prices = prices.with_columns(
            pl.col(schema_.price_date).dt.date().alias("trading_date"),
            ((pl.col(schema_.price_close) - pl.col(schema_.price_open)) / pl.col(schema_.price_open)).alias("return")
        )
        prices = prices.with_columns(
            pl.when(pl.col("return") > threshold_).then(0)
            .when(pl.col("return") < -threshold_).then(1)
            .otherwise(2).alias("NSI")
        )
        return prices.select(["trading_date", "ticker", "NSI"])

    @staticmethod
    def fine_tune_finbert(news_df: pl.LazyFrame, prices_df: pl.DataFrame, schema: Schema, s: Settings,
                          fine_tune_path_: Path, fine_tune_load_path_: Optional[Path]):
        # Compute NSI per day (small DataFrame)
        nsi_df = FinBertScorer.compute_nsi(prices_df, schema)
        Logger.info(f"Computed NSI for {len(nsi_df)} trading days.")

        dupe_stats = nsi_df.group_by("trading_date").agg(pl.count().alias("tickers_per_date"))
        Logger.info(f"Avg tickers per date: {dupe_stats['tickers_per_date'].mean()}")
        Logger.info(f"Max tickers per date: {dupe_stats['tickers_per_date'].max()}")

        # Lazily join with NSI (use nsi_df.lazy() to match types), drop nulls, select needed columns
        labeled_lazy = news_df.join(
            nsi_df.lazy(),
            left_on=["trading_date", schema.article_ticker],
            right_on=["trading_date", "ticker"],
            how="inner"
        )
        labeled_lazy = labeled_lazy.drop_nulls(subset=["NSI"])
        labeled_lazy = labeled_lazy.select([
            pl.col("text"),
            pl.col("NSI").cast(pl.Int8).alias("label")  # 0-2 fits in Int8
        ])

        # Compute num_examples cheaply before sinking (this is an aggregate, so no full materialization)
        num_examples = labeled_lazy.select(pl.count().alias("count")).collect()["count"][0]
        Logger.info(f"Number of labeled examples: {num_examples}")

        # Sink to temporary Parquet (streams without full collection)
        temp_file = Path("temp_labeled_data.parquet")
        Logger.info(f"Writing labeled data to temporary file: {temp_file}")
        labeled_lazy.sink_parquet(temp_file, compression='zstd', compression_level=10, maintain_order=False)
        Logger.info(f"Saved labeled data to temporary file: {temp_file}")

        # Load as non-streaming dataset (remove streaming=True)
        dataset = load_dataset("parquet", data_files=str(temp_file))["train"]

        # Split into train and test (e.g., 80/20 split)
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

        # Load tokenizer and model
        model_id = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)

        # Define tokenize function (applied on-the-fly)
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=s.max_length)

        # Apply tokenization (batched for efficiency)
        tokenized_train = split_dataset['train'].map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_eval = split_dataset['test'].map(tokenize_function, batched=True, remove_columns=["text"])

        # Set format to torch
        tokenized_train = tokenized_train.with_format("torch")
        tokenized_eval = tokenized_eval.with_format("torch")

        # Trainer setup
        training_args = TrainingArguments(
            output_dir=str(fine_tune_path_ / "finetuned_model_raw"),
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,  # Add this for eval batch size
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_steps=500,  # Evaluate every 500 steps
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # Optional: Use eval loss to select best model
        )

        # Calculate and set max_steps based on train examples
        steps_per_epoch = math.ceil(len(tokenized_train) / training_args.per_device_train_batch_size)
        training_args.max_steps = steps_per_epoch * training_args.num_train_epochs

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,  # Add this
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        trainer.train()

        # Save and cleanup
        fine_tune_path_str = str(fine_tune_path_)
        model.save_pretrained(fine_tune_path_str)
        tokenizer.save_pretrained(fine_tune_path_str)
        Logger.info(f"Fine-tuned FinBERT model and tokenizer saved to {fine_tune_path_str}")
        temp_file.unlink()
        Logger.info("Cleaned up temporary file.")

        return model, tokenizer
