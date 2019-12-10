
import pandas as pd

data = []

with open('data.txt') as f:
	for line in f.readlines():
		if not line:
			continue
		if line.startswith("[LABEL]"):
			label = line.strip("[LABEL]").strip()
		elif line.startswith("Alternatives:"):
			alternatives = line.strip("Alternatives:").strip().split(',')
			alternatives = [word.strip() for word in alternatives]
		elif line.startswith("Prefixes to above:"):
			prefixes = line.strip("Prefixes to above:").strip().split(',')
			prefixes = [word.strip() for word in prefixes]
			sentences = []
			for alternative in alternatives:
				data.append({'label': label, 'sentence': alternative})
				for prefix in prefixes:
					sentence = "{} {}".format(prefix, alternative)
					data.append({'label': label, 'sentence': sentence})

df = pd.DataFrame.from_dict(data)
print(df.head())
df.to_csv("bank_data.csv")