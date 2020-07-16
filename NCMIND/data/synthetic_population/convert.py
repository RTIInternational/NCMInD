import pandas as pd

df = pd.read_csv("NCMInD/data/synthetic_population/synthetic_population.csv")
df.to_csv("NCMInD/data/synthetic_population/synthetic_population.csv.xz")
