import pandas as pd
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [10, 20, 30],
    'c': [0.5, 1.5, 2.5],
    'd': [100, 200, 300]
})
print(df)
print(df.mean(axis=1))