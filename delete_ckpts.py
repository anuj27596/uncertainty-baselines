import os

# for seed in range(2,5):
#     root_dir = f"/home/anuj/troy/karm/outputs/rxrx1/vit/{seed}"
#     for each in os.listdir(os.path.join(root_dir, "checkpoints")):
#         if "74600" not in each:
#             print(os.path.join(root_dir, "checkpoints", each))
#             os.system(f'rm {os.path.join(root_dir, "checkpoints", each)}')
        

# for mr in [0.1,0.2,0.25,0.4,0.6,0.8]:
#     root_dir = f"/data5/scratch/karmpatel/outputs/isic/head_neck/vit_mae/mr_{mr}/0/checkpoints/"
#     for dir in os.listdir(root_dir):
#         if os.path.isdir(os.path.join(root_dir, dir)):
#             for each in os.listdir(os.path.join(root_dir, dir)):
#                 if "141900" not in each:
#                     print(os.path.join(root_dir, dir, each))
#                     os.system(f'rm -r {os.path.join(root_dir, dir, each)}')


for mr in [0.1,0.2,0.25,0.4,0.6,0.8]:
    root_dir = f"/data5/scratch/karmpatel/outputs/isic/head_neck/vit_mae_finetune/mr_{mr}/0/checkpoints/"
    for each in os.listdir(root_dir):
        if "142200" not in each:
            print(os.path.join(root_dir, each))
            os.system(f'rm -r {os.path.join(root_dir, each)}')

#os.listdir(root_dir)
