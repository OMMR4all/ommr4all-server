import glob
import os
import tempfile
from pathlib import Path

import loguru
import torch
from guppyocr.train_calamares import TrainingOpts, train_model
import shutil

path = "/home/alexanderh/Documents/data/"
data_paths = ["Pa_14819_gt", "n1_2_3", "mul2", "geesebook1"]

test_set = ["Pa_14819_gt_test", "n1_2_3_test", "mul2_test", "geesebook1_test"]

for ind, test_p in enumerate(test_set):

    datasets_paths = data_paths[:ind] + data_paths[ind + 1:] + test_set[:ind] + test_set[ind + 1:]
    output_path = os.path.join("/home/alexanderh/Documents/data/" + "_".join(datasets_paths))
    os.mkdir(output_path)
    with tempfile.TemporaryDirectory() as dirpath:
        os.mkdir(dirpath+"/train")
        for i in datasets_paths:
            path_n = os.path.join(path, i)
            for ti in glob.glob(f"{path_n}/*.png"):
                path_n = Path(ti)
                if path_n.with_suffix(".txt").exists():
                    shutil.copyfile(ti, os.path.join(dirpath+"/train", i+ os.path.basename(ti)))
                    shutil.copyfile(path_n.with_suffix(".txt"), os.path.join(dirpath+"/train", i+ os.path.basename(path_n.with_suffix(".txt"))))


        loguru.logger.info(f"Creating temporary train directory at {dirpath}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        training_opts = TrainingOpts(
            output=output_path,
            dataset=os.path.join(dirpath),
            # val_dataset=os.path.join(dirpath, "test"),
            model='',
            test_loaded_model=False,
            reader="plain",
            arch="crnn",
            gpu=True,
            worker=2,
            epoch=50,
            grad_clip=False
        )
        train_model(training_opts)