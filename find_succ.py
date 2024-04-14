import os

root_dir = ".logs/"

succ = []
for each in os.listdir(root_dir):
    if "histo_iw" in each:
        for file in os.listdir(os.path.join(root_dir, each)):
            if "succ" in file:
                succ.append(int(file.split("_")[-1].replace(".txt", "")))


print(set(succ))

total_jobs = list(range(48))
print(f"left: {[i for i in total_jobs if i not in succ]}")
