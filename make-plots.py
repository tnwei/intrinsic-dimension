import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Save results
df = pd.read_csv("./sample_output/fc-mnist-accuracy-20220427.csv", index_col=0)


tidydf = df.melt()
tidydf = tidydf.rename(columns={"variable": "num_id", "value": "acc"})

# Generate and save plot
fig = plt.figure(figsize=(14, 6))
sns.boxplot(data=tidydf, x="num_id", y="acc")
plt.xlabel("Intrinsic dimension")
plt.ylabel("Accuracy")
plt.title(
    "Accuracy of fully-connected network on MNIST, by constrained intrinsic dimension"
)
plt.savefig("./sample_output/fc-results-20220427.PNG", bbox_inches="tight")
plt.close(fig)



# Save results
df = pd.read_csv("./sample_output/conv-mnist-accuracy-20220427.csv", index_col=0)

tidydf = df.melt()
tidydf = tidydf.rename(columns={"variable": "num_id", "value": "acc"})

# Generate and save plot
fig = plt.figure(figsize=(14, 6))
sns.boxplot(data=tidydf, x="num_id", y="acc")
plt.xlabel("Intrinsic dimension")
plt.ylabel("Accuracy")
plt.title("Accuracy of conv network on MNIST, by constrained intrinsic dimension")
plt.savefig("./sample_output/conv-results-20220427.PNG", bbox_inches="tight")
plt.close(fig)
