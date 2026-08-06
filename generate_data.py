"""
Synthetic Data Generation Pipeline for SmartSales Analytics.
Generates realistic enterprise sales transactions with configurable distributions,
seasonal trends, customer behaviors, and controlled data quality anomalies.
"""
import argparse
import logging
from pathlib import Path
import sqlite3
from typing import List, Tuple
import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    CLEAN_DATA_PATH,
    DATABASE_PATH,
    NUM_CUSTOMERS,
    NUM_TRANSACTIONS,
    RANDOM_SEED,
    RAW_DATA_PATH
)
from src.data_cleaning import clean_data
from src.database import create_database
from src.logger import setup_logger

logger = setup_logger('generate_data')

def generate_synthetic_sales_data(
    num_records: int = NUM_TRANSACTIONS,
    num_customers: int = NUM_CUSTOMERS,
    seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate synthetic raw sales dataset with realistic distributions and intentional anomalies.

    Args:
        num_records: Number of transactions to generate
        num_customers: Number of unique customer IDs
        seed: Random seed for reproducibility

    Returns:
        DataFrame containing raw sales transactions
    """
    logger.info(f"Generating {num_records:,} synthetic records with seed {seed}")
    rng = np.random.default_rng(seed)

    regions = {
        'West': ['California', 'Washington', 'Oregon', 'Arizona', 'Nevada'],
        'East': ['New York', 'New Jersey', 'Massachusetts', 'Pennsylvania', 'Connecticut'],
        'Central': ['Texas', 'Illinois', 'Ohio', 'Michigan', 'Minnesota'],
        'South': ['Florida', 'Georgia', 'North Carolina', 'Virginia', 'Tennessee']
    }
    cities = {s: [f"{s} City A", f"{s} City B", f"{s} City C"] for states in regions.values() for s in states}

    product_catalog = [
        ('Electronics', 'Phones', 'Nova X Smartphone', 899.0, 620.0),
        ('Electronics', 'Laptops', 'ProBook 14 Laptop', 1299.0, 900.0),
        ('Electronics', 'Accessories', 'Wireless Keyboard', 79.0, 43.0),
        ('Electronics', 'Accessories', 'Noise Cancel Headphones', 149.0, 82.0),
        ('Electronics', 'Cameras', 'Mirrorless Camera', 999.0, 690.0),
        ('Furniture', 'Chairs', 'Ergo Office Chair', 249.0, 150.0),
        ('Furniture', 'Tables', 'Executive Desk', 499.0, 305.0),
        ('Furniture', 'Bookcases', 'Oak Bookcase', 289.0, 176.0),
        ('Furniture', 'Storage', 'Mobile Filing Cabinet', 159.0, 92.0),
        ('Office Supplies', 'Paper', 'Premium Copy Paper', 32.0, 16.0),
        ('Office Supplies', 'Binders', 'Professional Binder', 18.0, 8.0),
        ('Office Supplies', 'Art', 'Marker Set', 24.0, 10.0),
        ('Office Supplies', 'Labels', 'Shipping Label Pack', 22.0, 9.0),
        ('Office Supplies', 'Envelopes', 'Business Envelopes', 15.0, 6.0),
        ('Office Supplies', 'Supplies', 'Desk Organizer', 29.0, 12.0),
        ('Home & Kitchen', 'Appliances', 'Air Fryer', 139.0, 82.0),
        ('Home & Kitchen', 'Kitchen', 'Blender Pro', 119.0, 68.0),
        ('Home & Kitchen', 'Bedding', 'Comforter Set', 129.0, 76.0),
        ('Home & Kitchen', 'Decor', 'Smart LED Lamp', 69.0, 31.0),
        ('Home & Kitchen', 'Storage', 'Stackable Storage Box', 49.0, 22.0)
    ]

    customer_pool = [f"Customer {i:04d}" for i in range(1, num_customers + 1)]
    cust_weights = rng.lognormal(0.0, 1.0, len(customer_pool))
    cust_weights /= cust_weights.sum()

    region_names = list(regions.keys())
    sample_regions = rng.choice(region_names, num_records, p=[0.27, 0.24, 0.27, 0.22])
    sample_states = [rng.choice(regions[r]) for r in sample_regions]
    sample_cities = [rng.choice(cities[s]) for s in sample_states]
    sample_customers = rng.choice(customer_pool, num_records, p=cust_weights)

    start_date = pd.Timestamp('2022-01-01')
    random_days = rng.integers(0, 1461, num_records)
    sample_dates = start_date + pd.to_timedelta(random_days, unit='D')

    product_weights = np.array([
        0.07, 0.06, 0.07, 0.06, 0.04, 0.07, 0.05, 0.04, 0.04, 0.10,
        0.08, 0.05, 0.04, 0.05, 0.06, 0.05, 0.04, 0.04, 0.05, 0.04
    ])
    product_weights /= product_weights.sum()
    chosen_product_indices = rng.choice(len(product_catalog), num_records, p=product_weights)
    chosen_products = [product_catalog[i] for i in chosen_product_indices]

    quantities = rng.choice([1, 2, 3, 4, 5, 6, 7, 8], num_records, p=[0.32, 0.27, 0.17, 0.10, 0.06, 0.04, 0.025, 0.015])
    discounts = np.clip(rng.beta(2.0, 12.0, num_records) * 0.55, 0.0, 0.55)

    base_prices = np.array([p[3] for p in chosen_products])
    base_costs = np.array([p[4] for p in chosen_products])

    sales = np.round(base_prices * quantities * (1.0 - discounts) * rng.normal(1.0, 0.035, num_records), 2)
    costs = np.round(base_costs * quantities * rng.normal(1.0, 0.025, num_records), 2)
    profits = np.round(sales - costs, 2)

    order_ids = [f"ORD-{i:06d}" for i in range(1, num_records + 1)]

    df = pd.DataFrame({
        'Order ID': order_ids,
        'Order Date': sample_dates,
        'Customer': sample_customers,
        'Product': [p[2] for p in chosen_products],
        'Category': [p[0] for p in chosen_products],
        'Sub-Category': [p[1] for p in chosen_products],
        'Quantity': quantities,
        'Sales': sales,
        'Discount': discounts,
        'Cost': costs,
        'Profit': profits,
        'Region': sample_regions,
        'State': sample_states,
        'City': sample_cities,
        'Payment Mode': rng.choice(['Credit Card', 'Debit Card', 'UPI', 'Cash', 'PayPal'], num_records, p=[0.35, 0.20, 0.22, 0.08, 0.15]),
        'Shipping Mode': rng.choice(['Standard Class', 'Second Class', 'First Class', 'Same Day'], num_records, p=[0.50, 0.25, 0.17, 0.08])
    })

    anomaly_indices = rng.choice(df.index, size=650, replace=False)
    df.loc[anomaly_indices[:120], 'Customer'] = None
    df.loc[anomaly_indices[120:220], 'City'] = None
    df.loc[anomaly_indices[220:320], 'Discount'] = None
    df.loc[anomaly_indices[320:380], 'Payment Mode'] = df.loc[anomaly_indices[320:380], 'Payment Mode'].str.lower()
    df.loc[anomaly_indices[380:440], 'Category'] = df.loc[anomaly_indices[380:440], 'Category'].str.upper()
    df.loc[anomaly_indices[500:550], 'Sales'] *= 8
    df.loc[anomaly_indices[550:600], 'Profit'] *= -4

    duplicates = df.sample(100, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)

    logger.info(f"Generated raw dataset: {len(df):,} rows (including {len(duplicates)} duplicates)")
    return df

def main() -> None:
    """CLI execution for generating synthetic datasets and populating SQLite database."""
    parser = argparse.ArgumentParser(description="Generate synthetic SmartSales enterprise dataset.")
    parser.add_argument('--records', type=int, default=NUM_TRANSACTIONS, help="Number of records to generate")
    parser.add_argument('--customers', type=int, default=NUM_CUSTOMERS, help="Number of unique customers")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Random generator seed")
    parser.add_argument('--raw-output', type=str, default=str(RAW_DATA_PATH), help="Raw CSV output path")
    parser.add_argument('--clean-output', type=str, default=str(CLEAN_DATA_PATH), help="Clean CSV output path")
    parser.add_argument('--db-path', type=str, default=str(DATABASE_PATH), help="Target SQLite database path")
    args = parser.parse_args()

    raw_df = generate_synthetic_sales_data(num_records=args.records, num_customers=args.customers, seed=args.seed)
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Raw dataset successfully saved to: {raw_path}")

    clean_df = clean_data(raw_df)
    clean_path = Path(args.clean_output)
    clean_df.to_csv(clean_path, index=False)
    logger.info(f"Cleaned dataset successfully saved to: {clean_path}")

    db_path = Path(args.db_path)
    create_database(clean_df, db_path=db_path)
    logger.info(f"Database successfully populated at: {db_path}")
    print(f"Dataset Generation Complete: Raw={len(raw_df):,}, Clean={len(clean_df):,}, Database={db_path}")

if __name__ == '__main__':
    main()
