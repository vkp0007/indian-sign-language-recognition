import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results_log.csv")
df["Detection_%"] = df["Ratio"] * 100

# 🔥 Sort classes (important for readability)
class_summary = df.groupby("Label")["Detection_%"].mean().reset_index()
class_summary = class_summary.sort_values(by="Detection_%", ascending=False)

plt.figure(figsize=(12, 5), dpi=300)

bars = plt.bar(class_summary["Label"], class_summary["Detection_%"], edgecolor='black')

# 🔥 Add average line (very useful for paper)
avg = class_summary["Detection_%"].mean()
plt.axhline(avg, linestyle='--', linewidth=1)
plt.text(len(class_summary)-1, avg+1, f"Avg: {avg:.1f}%", ha='right')

# 🔥 Optional: annotate top few bars (avoid clutter)
for i, v in enumerate(class_summary["Detection_%"]):
    if i < 5:  # only top 5
        plt.text(i, v + 1, f"{v:.1f}", ha='center', fontsize=8)

plt.xlabel("Sign Class", fontsize=12)
plt.ylabel("Detection Accuracy (%)", fontsize=12)
plt.title("Class-wise Detection Performance", fontsize=13)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("class_performance.pdf")
plt.show()