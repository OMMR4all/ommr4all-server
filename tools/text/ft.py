import glob
import os
import shutil
import tempfile
from pathlib import Path

import loguru
import torch
from guppyocr.train_calamares import TrainingOpts, train_model

data = [("n1_2_3_mul2_geesebook1_n1_2_3_test_mul2_test_geesebook1_test", "Pa_14819_gt"),
 ("Pa_14819_gt_mul2_geesebook1_Pa_14819_gt_test_mul2_test_geesebook1_test", "n1_2_3"),
 ("Pa_14819_gt_n1_2_3_geesebook1_Pa_14819_gt_test_n1_2_3_test_geesebook1_test", "mul2"),
 ("Pa_14819_gt_n1_2_3_mul2_Pa_14819_gt_test_n1_2_3_test_mul2_test", "geesebook1")]

finetune_síze = 20, 50, 100, 500

for ind, test_p in enumerate(data):
    for i in finetune_síze:
        folder = test_p[0]
        dataset = test_p[1]
        output_path = os.path.join("/home/alexanderh/Documents/data/",folder, "finetunedon_"+ dataset+"_" + str(i))
        os.mkdir(output_path)
        with tempfile.TemporaryDirectory() as dirpath:
            os.mkdir(dirpath+"/train")
            os.mkdir(dirpath+"/test")

            path_n = os.path.join("/home/alexanderh/Documents/data/", dataset)
            files = glob.glob(f"{path_n}/*.png")
            filted_images = []
            for j in files:
                if "debug" in j:
                    pass
                else:
                    filted_images.append(j)
            from random import shuffle
            shuffle(filted_images)
            filted_images = filted_images[:i]
            train = filted_images[:int(len(filted_images) * 0.8)]
            test = filted_images[int(len(filted_images) * 0.8):]
            for ti in train:

                path_n = Path(ti)


                if path_n.with_suffix(".txt").exists():
                    shutil.copyfile(ti, os.path.join(dirpath+"/train", os.path.basename(ti)))
                    shutil.copyfile(path_n.with_suffix(".txt"), os.path.join(dirpath+"/train", os.path.basename(path_n.with_suffix(".txt"))))
            for ti in test:

                path_n = Path(ti)


                if path_n.with_suffix(".txt").exists():
                    shutil.copyfile(ti, os.path.join(dirpath+"/test", os.path.basename(ti)))
                    shutil.copyfile(path_n.with_suffix(".txt"), os.path.join(dirpath+"/test", os.path.basename(path_n.with_suffix(".txt"))))

            loguru.logger.info(f"Creating temporary train directory at {dirpath}")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            training_opts = TrainingOpts(
                output=output_path,
                dataset=os.path.join(dirpath),
                # val_dataset=os.path.join(dirpath, "test"),
                model=os.path.join("/home/alexanderh/Documents/data/",folder, "model_best.pth"),
                test_loaded_model=False,
                reader="plain",
                arch="crnn",
                gpu=True,
                worker=2,
                epoch=50,
                grad_clip=False,
                augment=True,
            )
            train_model(training_opts)