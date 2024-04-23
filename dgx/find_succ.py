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

# from cmds.txt
run_file = ".logs/histo_iw_first_five_g6/cmds.txt"
with open(run_file, "r") as fp:
    runs = fp.readlines()
runs = list(filter(lambda x: "output_dir" in x, runs))
run_ids = {}
for ind, run in enumerate(runs):
    run_ids[run.split("outputs")[-1].strip()[:-1].strip()] = ind


# from select from local file
sweep_file = "txt_files/histo_iw.txt"
with open(sweep_file, "r") as fp:
    sweeps = fp.readlines()
sweeps = [each.split("outputs")[-1].split(":")[0].strip() for each in list(filter(lambda x: ":" in x, sweeps))]
print(f"{len(sweeps)} sweeps found")

# missing sweeps
missing_ids = []
for run in run_ids:
    if run not in sweeps:
        print(run)
        missing_ids.append(run_ids[run])
print("missing ids")
print(missing_ids)

print()