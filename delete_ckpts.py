import os

# for seed in range(2,5):
#     root_dir = f"/home/anuj/troy/karm/outputs/rxrx1/vit/{seed}"
#     for each in os.listdir(os.path.join(root_dir, "checkpoints")):
#         if "74600" not in each:
#             print(os.path.join(root_dir, "checkpoints", each))
#             os.system(f'rm {os.path.join(root_dir, "checkpoints", each)}')
        

for seed in range(3,4):
    root_dir = f"/home/anuj/troy/karm/outputs/rxrx1/vit/{seed}"
    for dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, dir)):
            for each in os.listdir(os.path.join(root_dir, dir)):
                if "74600" not in each:
                    print(os.path.join(root_dir, dir, each))
                    os.system(f'rm -r {os.path.join(root_dir, dir, each)}')

#os.listdir(root_dir)
