from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, pipeline)
from typing import List
import numpy as np
import pandas as pd
import torch

from components.article_preprocessor import ArticlePreprocessor
from components.types import Schema, Settings

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
    def __init__(self, batch_size: int, max_length: int, model=None, tokenizer=None):
        device = 0 if torch.cuda.is_available() else -1
        if model is None:
            model_id = "ProsusAI/finbert"
            mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
            tok = AutoTokenizer.from_pretrained(model_id)
        else:
            mdl = model
            # If caller didn’t pass a tokenizer, fall back to matching one
            tok = tokenizer or AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.pipe = pipeline(
            "text-classification",
            model=mdl,
            tokenizer=tok,
            device=device,
            return_all_scores=True,
            truncation=True
        )
        if torch.cuda.is_available() and hasattr(torch, "compile"):
            self.pipe.model = torch.compile(self.pipe.model) # Compile the model
        self.batch_size = batch_size
        # clamp to tokenizer's configured limit
        try:
            self.max_length = min(max_length, getattr(self.pipe.tokenizer, "model_max_length", max_length) or max_length)
        except Exception:
            self.max_length = max_length
    def score_texts(self, texts: List[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["p_pos", "p_neg", "p_neu", "sentiment_score"])
        results = []
        self.pipe.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="FinBERT", unit="batch"):
                batch = texts[i:i + self.batch_size]
                inputs = self.pipe.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.pipe.device) for k, v in inputs.items()}
                with autocast("cuda"): # Enable mixed precision
                    outputs = self.pipe.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                for prob in probs:
                    p_pos, p_neg, p_neu = prob
                    results.append((p_pos, p_neg, p_neu, p_pos - p_neg))
        return pd.DataFrame(results, columns=["p_pos", "p_neg", "p_neu", "sentiment_score"])

    @staticmethod
    def compute_nsi(prices_df: pd.DataFrame, schema: Schema, threshold: float = 0.01) -> pd.DataFrame:
        prices = prices_df.copy()
        prices["trading_date"] = pd.to_datetime(prices[schema.price_date]).dt.date
        prices["return"] = (prices[schema.price_close] - prices[schema.price_open]) / prices[schema.price_open]
        prices["NSI"] = np.where(prices["return"] > threshold, 0,
                                np.where(prices["return"] < -threshold, 1, 2)) # Map: pos=0, neg=1, neu=2 to match FinBERT labels
        return prices[["trading_date", "NSI"]]

    @staticmethod
    def fine_tune_finbert(news_df: pd.DataFrame, prices_df: pd.DataFrame, schema: Schema, s: Settings,
                        prep: ArticlePreprocessor):
        # Compute NSI per day
        nsi_df = FinBertScorer.compute_nsi(prices_df, schema)
        # Map news to trading_date and label with NSI
        news_df["trading_date"] = prep.compute_trading_date(news_df["published_utc"]).dt.date
        labeled_df = pd.merge(news_df, nsi_df, on="trading_date", how="inner")
        labeled_df = labeled_df.dropna(subset=["NSI"])
        texts = prep.build_text(labeled_df).tolist()
        labels = labeled_df["NSI"].astype(int).tolist() # 0: pos, 1: neg, 2: neu
        # Load base FinBERT
        model_id = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3)
        # Dataset
        train_dataset = NSIDataset(texts, labels, tokenizer, s.max_length)
        # Trainer
        training_args = TrainingArguments(
            output_dir="./finbert_finetuned",
            num_train_epochs=3, # Paper suggests task-specific fine-tuning; adjust as needed
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="no",
            load_best_model_at_end=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        trainer.train()
        # Save and return fine-tuned model
        model.save_pretrained("./finbert_finetuned")
        tokenizer.save_pretrained("./finbert_finetuned")
        return AutoModelForSequenceClassification.from_pretrained("./finbert_finetuned"), tokenizer
