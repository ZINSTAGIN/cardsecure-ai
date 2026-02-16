"""
Generate Test Data - 98% Fraud
"""

import pandas as pd
import numpy as np

np.random.seed(123)

samples = []

# 98 Fraud transactions
for _ in range(98):
    samples.append({
        'Time': np.random.uniform(0, 172800),
        'V1': np.random.uniform(-4.0, -2.0),
        'V2': np.random.uniform(2.0, 4.5),
        'V3': np.random.uniform(-3.0, -1.0),
        'V4': np.random.uniform(2.5, 5.5),
        'V5': np.random.uniform(-3.0, -1.0),
        'V6': np.random.uniform(-2.5, -0.5),
        'V7': np.random.uniform(2.0, 4.5),
        'V8': np.random.uniform(-4.0, -1.5),
        'V9': np.random.uniform(0.5, 2.5),
        'V10': np.random.uniform(-3.0, -0.5),
        'V11': np.random.uniform(-2.0, 0.0),
        'V12': np.random.uniform(-2.0, 1.0),
        'V13': np.random.uniform(-1.5, 0.5),
        'V14': np.random.uniform(-6.0, -3.0),
        'V15': np.random.uniform(-1.0, 1.0),
        'V16': np.random.uniform(-4.0, -1.0),
        'V17': np.random.uniform(-5.0, -2.0),
        'V18': np.random.uniform(-2.0, 0.0),
        'V19': np.random.uniform(0.0, 2.0),
        'V20': np.random.uniform(0.0, 1.0),
        'V21': np.random.uniform(0.0, 1.0),
        'V22': np.random.uniform(-1.5, 0.0),
        'V23': np.random.uniform(-0.5, 0.5),
        'V24': np.random.uniform(-1.0, 0.0),
        'V25': np.random.uniform(0.0, 1.0),
        'V26': np.random.uniform(-0.5, 0.5),
        'V27': np.random.uniform(0.0, 0.5),
        'V28': np.random.uniform(-0.5, 0.0),
        'Amount': np.random.uniform(500, 5000)
    })

# 2 Legitimate transactions
for _ in range(2):
    samples.append({
        'Time': np.random.uniform(0, 172800),
        'V1': np.random.uniform(-1.0, 1.0),
        'V2': np.random.uniform(-1.0, 1.0),
        'V3': np.random.uniform(-0.5, 2.0),
        'V4': np.random.uniform(-1.0, 1.0),
        'V5': np.random.uniform(-1.0, 1.0),
        'V6': np.random.uniform(-1.0, 1.0),
        'V7': np.random.uniform(-1.0, 1.0),
        'V8': np.random.uniform(-0.5, 0.5),
        'V9': np.random.uniform(-1.0, 1.0),
        'V10': np.random.uniform(-1.0, 1.0),
        'V11': np.random.uniform(-1.5, 1.5),
        'V12': np.random.uniform(-1.5, 1.5),
        'V13': np.random.uniform(-1.5, 0.5),
        'V14': np.random.uniform(-1.0, 2.0),
        'V15': np.random.uniform(-1.0, 1.0),
        'V16': np.random.uniform(-1.0, 1.0),
        'V17': np.random.uniform(-1.0, 1.0),
        'V18': np.random.uniform(-1.0, 1.0),
        'V19': np.random.uniform(-1.0, 1.0),
        'V20': np.random.uniform(-0.5, 0.5),
        'V21': np.random.uniform(-0.5, 0.5),
        'V22': np.random.uniform(-1.0, 1.0),
        'V23': np.random.uniform(-0.5, 0.5),
        'V24': np.random.uniform(-0.5, 0.5),
        'V25': np.random.uniform(-0.5, 0.5),
        'V26': np.random.uniform(-0.5, 0.5),
        'V27': np.random.uniform(-0.3, 0.3),
        'V28': np.random.uniform(-0.3, 0.3),
        'Amount': np.random.uniform(1, 300)
    })

np.random.shuffle(samples)

df = pd.DataFrame(samples)
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
df = df[columns].round(2)

df.to_csv('fraud_test_data.csv', index=False)

print("=" * 50)
print("✅ Created: fraud_test_data.csv")
print(f"   Total rows: 100")
print(f"   Fraud: 98 (98%)")
print(f"   Legitimate: 2 (2%)")
print("=" * 50)