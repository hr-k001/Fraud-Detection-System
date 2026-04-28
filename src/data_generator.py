"""
Fake Transaction Data Generator

Generates synthetic transaction data for testing and training the fraud detection system.
Creates realistic patterns with various transaction types, amounts, and user behaviors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import random


class FakeTransactionGenerator:
    """Generate synthetic transaction data with realistic patterns."""

    def __init__(self, seed: int = 42):
        """Initialize the generator with optional random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)

        # Product categories
        self.products = ['W', 'H', 'S', 'C', 'R']  # Wallet, Home, Shopping, Cards, Revenue
        
        # Device types and platforms
        self.device_types = ['desktop', 'mobile', 'tablet']
        self.os_types = ['Windows', 'MacOS', 'Android', 'iOS', 'Linux']
        self.browsers = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Opera']
        
        # Geographic data
        self.countries = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP', 'IN', 'BR', 'MX',
                         'ES', 'IT', 'NL', 'CH', 'SE', 'NO', 'DK', 'BE', 'AT', 'PL']
        self.states = {
            'US': ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
            'GB': ['ENG', 'SCT', 'WLS', 'NIR'],
            'CA': ['ON', 'BC', 'AB', 'MB', 'NS'],
        }

        # Merchants
        self.merchant_names = [
            'Amazon', 'Walmart', 'Target', 'Best Buy', 'Apple',
            'Starbucks', 'McDonald', 'Uber', 'Lyft', 'Netflix',
            'Spotify', 'Adobe', 'Microsoft', 'Google Cloud', 'AWS'
        ]

    def generate_transactions(self, n_samples: int = 10000, fraud_ratio: float = 0.1) -> pd.DataFrame:
        """
        Generate synthetic transaction dataset.
        
        Args:
            n_samples: Total number of transactions to generate
            fraud_ratio: Ratio of fraudulent transactions (0.0-1.0)
        
        Returns:
            DataFrame with synthetic transaction data
        """
        n_fraud = int(n_samples * fraud_ratio)
        n_legitimate = n_samples - n_fraud

        # Generate legitimate transactions
        legitimate_txns = self._generate_legitimate_transactions(n_legitimate)
        
        # Generate fraudulent transactions
        fraudulent_txns = self._generate_fraudulent_transactions(n_fraud)
        
        # Combine and shuffle
        df = pd.concat([legitimate_txns, fraudulent_txns], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df

    def _generate_legitimate_transactions(self, n: int) -> pd.DataFrame:
        """Generate legitimate transaction records."""
        data = {
            'TransactionID': [f'TXN_{i:08d}' for i in range(n)],
            'TransactionAmt': np.random.lognormal(3.5, 1.5, n),  # Log-normal distribution
            'ProductCD': np.random.choice(self.products, n, p=[0.3, 0.25, 0.25, 0.15, 0.05]),
            'isFraud': np.zeros(n, dtype=int),
            'DayOfWeek': np.random.randint(0, 7, n),
            'Hour': np.random.randint(0, 24, n),
            'CardType': np.random.choice(['credit', 'debit'], n, p=[0.6, 0.4]),
            'DeviceType': np.random.choice(self.device_types, n),
            'OS': np.random.choice(self.os_types, n),
            'Browser': np.random.choice(self.browsers, n),
            'Country': np.random.choice(self.countries, n),
            'State': self._generate_states(n),
            'Merchant': np.random.choice(self.merchant_names, n),
            'EmailDomain': self._generate_email_domains(n),
            'Distance_km': np.random.exponential(500, n),  # Distance in km
            'DaysSincePreviousTxn': np.random.exponential(10, n),
            'NumPreviousTxns': np.random.poisson(20, n),
        }
        
        df = pd.DataFrame(data)
        
        # Round amounts to 2 decimal places
        df['TransactionAmt'] = df['TransactionAmt'].round(2)
        df['Distance_km'] = df['Distance_km'].round(2)
        df['DaysSincePreviousTxn'] = df['DaysSincePreviousTxn'].round(2)
        
        return df

    def _generate_fraudulent_transactions(self, n: int) -> pd.DataFrame:
        """Generate fraudulent transaction records with suspicious patterns."""
        data = {
            'TransactionID': [f'TXN_FRAUD_{i:08d}' for i in range(n)],
            'TransactionAmt': np.random.choice(
                [np.random.uniform(1000, 5000) for _ in range(n)],  # Higher amounts
                n
            ),
            'ProductCD': np.random.choice(self.products, n, p=[0.5, 0.2, 0.15, 0.1, 0.05]),  # Biased toward W
            'isFraud': np.ones(n, dtype=int),
            'DayOfWeek': np.random.randint(0, 7, n),
            'Hour': np.random.choice(range(0, 24), n),  # Unusual hours
            'CardType': np.random.choice(['credit', 'debit'], n, p=[0.8, 0.2]),  # More credit cards
            'DeviceType': np.random.choice(self.device_types, n, p=[0.2, 0.7, 0.1]),  # Mobile biased
            'OS': np.random.choice(self.os_types, n),
            'Browser': np.random.choice(self.browsers, n),
            'Country': np.random.choice(self.countries, n),
            'State': self._generate_states(n),
            'Merchant': np.random.choice(self.merchant_names, n),
            'EmailDomain': self._generate_email_domains(n),
            'Distance_km': np.random.exponential(2000, n),  # Unusual distances
            'DaysSincePreviousTxn': np.random.exponential(2, n),  # Rapid transactions
            'NumPreviousTxns': np.random.poisson(5, n),  # Lower previous transactions
        }
        
        df = pd.DataFrame(data)
        
        # Round amounts to 2 decimal places
        df['TransactionAmt'] = df['TransactionAmt'].round(2)
        df['Distance_km'] = df['Distance_km'].round(2)
        df['DaysSincePreviousTxn'] = df['DaysSincePreviousTxn'].round(2)
        
        return df

    def _generate_states(self, n: int) -> List[str]:
        """Generate states based on country."""
        states = []
        for _ in range(n):
            country = np.random.choice(self.countries)
            if country in self.states:
                states.append(np.random.choice(self.states[country]))
            else:
                states.append('XX')  # Unknown state
        return states

    def _generate_email_domains(self, n: int) -> List[str]:
        """Generate email domains."""
        domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 
                  'protonmail.com', 'icloud.com', 'mail.com', 'aol.com']
        return [np.random.choice(domains) for _ in range(n)]

    def generate_temporal_sequences(self, n_users: int = 1000, 
                                   transactions_per_user: int = 50) -> pd.DataFrame:
        """
        Generate transactions with temporal sequences for a user population.
        More realistic for modeling user behavior patterns.
        
        Args:
            n_users: Number of unique users
            transactions_per_user: Average transactions per user
        
        Returns:
            DataFrame with temporal transaction sequences
        """
        all_txns = []
        base_time = datetime.now() - timedelta(days=365)

        for user_id in range(n_users):
            # Random number of transactions for this user
            n_txns = np.random.poisson(transactions_per_user)
            is_fraudster = np.random.random() < 0.05  # 5% fraudsters
            
            for txn_idx in range(n_txns):
                # Time progression
                days_offset = np.random.exponential(5)
                txn_time = base_time + timedelta(days=days_offset)
                
                # Transaction amount
                if is_fraudster and np.random.random() < 0.3:
                    # Fraudster: occasional high-value transactions
                    amount = np.random.uniform(2000, 10000)
                else:
                    # Normal user
                    amount = np.random.lognormal(3.5, 1.5)
                
                txn = {
                    'UserID': f'USER_{user_id:08d}',
                    'TransactionID': f'TXN_{user_id:08d}_{txn_idx:06d}',
                    'DateTime': txn_time,
                    'TransactionAmt': round(amount, 2),
                    'ProductCD': np.random.choice(self.products),
                    'isFraud': 1 if (is_fraudster and np.random.random() < 0.3) else 0,
                    'CardType': np.random.choice(['credit', 'debit']),
                    'DeviceType': np.random.choice(self.device_types),
                    'Country': np.random.choice(self.countries),
                    'Merchant': np.random.choice(self.merchant_names),
                }
                
                all_txns.append(txn)
        
        df = pd.DataFrame(all_txns)
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        return df


def main():
    """Generate and save sample datasets."""
    generator = FakeTransactionGenerator(seed=42)
    
    # Generate basic dataset
    print("Generating basic transaction dataset...")
    df_basic = generator.generate_transactions(n_samples=10000, fraud_ratio=0.1)
    df_basic.to_csv('/Users/mansidaksingh/capstone_project/Fraud-Detection-System/data/synthetic_transactions.csv', index=False)
    print(f"Generated {len(df_basic)} transactions")
    print(f"Fraud ratio: {df_basic['isFraud'].mean():.2%}")
    print(df_basic.head())
    print("\n" + "="*50 + "\n")
    
    # Generate temporal sequences
    print("Generating temporal transaction sequences...")
    df_temporal = generator.generate_temporal_sequences(n_users=500, transactions_per_user=30)
    df_temporal.to_csv('/Users/mansidaksingh/capstone_project/Fraud-Detection-System/data/temporal_transactions.csv', index=False)
    print(f"Generated {len(df_temporal)} transactions for {df_temporal['UserID'].nunique()} users")
    print(f"Fraud ratio: {df_temporal['isFraud'].mean():.2%}")
    print(df_temporal.head())


if __name__ == '__main__':
    main()
