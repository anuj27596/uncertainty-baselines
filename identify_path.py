import os
prefix_local_dirs = [
	"/home/anuj/troy/gub-og/outputs/isic/new_sweep_results",
	"/home/anuj/troy/gub-og/outputs/vit/new_sweep_results",
	"/home/anuj/troy/gub-og/outputs/chest_xray/new_sweep_results"
 ]

for dataset_path in prefix_local_dirs:
    print(dataset_path)
    for model in os.listdir(dataset_path):
        for lr in os.listdir(os.path.join(dataset_path, model)):
            for hparm in os.listdir(os.path.join(dataset_path, model, lr)):
                # print(os.path.join(dataset_path, model, lr, hparm))
                if len(os.listdir(os.path.join(dataset_path, model, lr, hparm))) == 6:
                    # print("****************")
                    print(os.path.join(dataset_path, model, lr, hparm))
                    # print("****************")
    print()
        