"""Feature engineering: RFM extraction, scaling (placeholder)"""
import pandas as pd
import numpy as np
from datetime import timedelta


def compute_rfm(
	df,
	customer_id_col="CustomerID",
	date_col="InvoiceDate",
	quantity_col="Quantity",
	price_col="UnitPrice",
	snapshot_date=None,
):
	"""Compute RFM (Recency, Frequency, Monetary) scores for customers.

	Expects `date_col` to be datetime dtype or parsable by pd.to_datetime.
	Returns a DataFrame indexed by customer id with columns ['Recency','Frequency','Monetary'].
	"""
	if date_col not in df.columns:
		raise KeyError(f"{date_col} not in DataFrame")

	df = df.copy()
	if not np.issubdtype(df[date_col].dtype, np.datetime64):
		df[date_col] = pd.to_datetime(df[date_col])

	if snapshot_date is None:
		snapshot_date = df[date_col].max() + timedelta(days=1)

	df["_amount"] = df[quantity_col] * df[price_col]

	grouped = df.groupby(customer_id_col).agg(
		Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
		Frequency=(date_col, "count"),
		Monetary=("_amount", "sum"),
	)

	grouped = grouped.rename(columns={"Frequency": "Frequency", "Monetary": "Monetary"})
	return grouped


def scale_features(df):
	"""Scale numeric features to zero mean and unit variance.

	Accepts a pandas DataFrame or 2D numpy array. Returns a DataFrame when input is DataFrame.
	"""
	try:
		from sklearn.preprocessing import StandardScaler

		scaler = StandardScaler()
		if isinstance(df, pd.DataFrame):
			cols = df.columns
			arr = scaler.fit_transform(df.values)
			return pd.DataFrame(arr, index=df.index, columns=cols)
		else:
			return scaler.fit_transform(df)
	except Exception:
		# fallback: numpy-based standardization
		arr = np.asarray(df, dtype=float)
		mean = arr.mean(axis=0)
		std = arr.std(axis=0)
		std[std == 0] = 1.0
		scaled = (arr - mean) / std
		if isinstance(df, pd.DataFrame):
			return pd.DataFrame(scaled, index=df.index, columns=df.columns)
		return scaled


__all__ = ["compute_rfm", "scale_features"]
