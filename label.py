import os

DATASET_PATH = "dataset_isl"

labels = {}
label_index = 0

# Build label dictionary
for word in sorted(os.listdir(DATASET_PATH)):
    folder_path = os.path.join(DATASET_PATH, word)

    if os.path.isdir(folder_path):
        labels[word] = label_index
        label_index += 1

print("Labels mapping:")
print(labels)

print("\nDataset structure:\n")

# Show folders and files
for root, dirs, files in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")

    sub_indent = ' ' * 4 * (level + 1)
    for file in files:
        print(f"{sub_indent}{file}")