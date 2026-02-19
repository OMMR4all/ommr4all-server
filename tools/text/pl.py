import glob
import os
from pathlib import Path

import cv2
import torch
from guppyocr.predict_pxml import preprocess_image
from guppyocr.api import GuppyOCR, InvalidInputImage
from omr.steps.text.guppy.predictor import resize_with_pad, GreedyDecoder, DecoderOutput
path = "/home/alexanderh/Documents/data/"
data = [("n1_2_3_mul2_geesebook1_n1_2_3_test_mul2_test_geesebook1_test", "Pa_14819_gt_test"),
 ("Pa_14819_gt_mul2_geesebook1_Pa_14819_gt_test_mul2_test_geesebook1_test", "n1_2_3_test"),
 ("Pa_14819_gt_n1_2_3_geesebook1_Pa_14819_gt_test_n1_2_3_test_geesebook1_test", "mul2_test"),
 ("Pa_14819_gt_n1_2_3_mul2_Pa_14819_gt_test_n1_2_3_test_mul2_test", "geesebook1_test")]

for i in data:
    model = os.path.join(path, i[0]) + "/model_best.pth"
    images = glob.glob(os.path.join(path, i[1]) +"/*.png")

    for img_path in images:
        image =cv2.imread(img_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = GuppyOCR.load_model(model, device)
        network.model.eval()
        img, ratio = preprocess_image(image, network.mc.mconfig.Width, network.mc.mconfig.Height)

        img_t, sizes = resize_with_pad(image, (network.mc.mconfig.Width, network.mc.mconfig.Height),
                                       (255, 255, 255))
        img = img[None, :, :, :]
        with torch.no_grad():
            img = img.to(device)
            prediction, _, _, _ = network.model.forward(img, None)

            net_out = prediction[0].cpu().numpy()
            alphabet = [0] * len(network.mc.mconfig.id2char)
            for k, v in network.mc.mconfig.id2char.items():
                alphabet[k] = v
            # from fast_ctc_decode import viterbi_search

            # seq, _ = viterbi_search(net_out, alphabet)
            g_decoder = GreedyDecoder(alphabet)
            sentence: DecoderOutput = g_decoder.decode(net_out, True, debug_img=img_t, pad=sizes[1], debug=False)
            with open(Path(img_path).with_suffix(".lstm.txt"), "w") as f:
                f.write(sentence.decoded_string)