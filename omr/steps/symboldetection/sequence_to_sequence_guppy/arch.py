import os
from pathlib import Path
import random
from typing import Union, BinaryIO, IO, Tuple
import math
import dataclasses
import tqdm
import torch
from adabelief_pytorch import AdaBelief
from fast_ctc_decode import viterbi_search
from guppyocr.dataset import dataset
from guppyocr.model.charmap import make_charmap_from_readers
from guppyocr.model.model import ModelParameters, adjust_voc_model
from guppyocr.model.network import CalamariNetv2, CRNN, STNCRNN, TrCRNN, BidirectionalLSTM
from guppyocr.train_calamares import test_model, make_readers, TrainingOpts, TrainingResult
from torch import nn
import numpy as np
import editdistance

def decode_ctc(pred, id2char) -> str:
    gt_chars = []
    pred_chars = []
    alphabet = [0] * len(id2char)
    net_out = pred.cpu().numpy()
    # print(net_out.shape)
    for k, v in id2char.items():
        alphabet[k] = v
    # print("Alph: ", len(alphabet))
    # print(alphabet)
    seq, _ = viterbi_search(net_out, alphabet)
    seq_out = []
    for x in seq:
        if x == "PAD" or x == "UNK" or x == "END":
            continue
        seq_out.append(x)
    return "".join(seq)


def decode_gt(gt, id2char) -> str:
    gt_chars = []
    for x in gt:
        trans = id2char[int(x)]
        if trans == "PAD" or trans == "UNK" or trans == "END":
            continue
        gt_chars.append(trans)
    return "".join(gt_chars)





class MetricAggregator:
    def __init__(self):
        pass

class CRNNSymbol(nn.Module):

    def __init__(self, Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device, leakyRelu=False, pre_lstm_dropout: float = 0.0, inter_lstm_dropout: float = 0.0):
        super(CRNNSymbol, self).__init__()

        self.channel = Channel
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.output_classes = output_classes
        self.seq_len = seq_len
        #self.dropout = dropout

        #assert opt['imgH'] % 16 == 0, 'imgH has to be a multiple of 16'
        assert self.feature_height % 16 == 0


        ks = [3, 3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = self.channel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        # (((W - K + 2P)/S) + 1)
        # (48 - 3 + 2) / 1 +

        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  #

        convRelu(4)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16

        convRelu(5, True)
        convRelu(6)
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((4, 2), (2, 1), (0, 1)))  # 512x2x16

        convRelu(7, True)  # 512x1x16
        if pre_lstm_dropout > 0:
            cnn.add_module('dropout0', nn.Dropout(pre_lstm_dropout))
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 128, 128),
            nn.Dropout(inter_lstm_dropout) if inter_lstm_dropout > 0 else nn.Identity(),
            BidirectionalLSTM(128, 128, self.output_classes))
        #print(self.cnn)


    def forward(self, input, _):
        # conv features

        conv = self.cnn(input)
        b, c, h, w = conv.size()
        #print(f"Conv Size: {b} {c} {h} {w}")
        assert h == 1, f"the height of conv must be 1, is: {h}"
        conv = conv.squeeze(2)
        #print(conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        #print(conv.size())
        # rnn features
        output = self.rnn(conv)
        #print(output.size())
        output = output.transpose(1,0) #Tbh to bth
        #print(output.size())
        return output, None, None, None


class ModelConfiguration:
    def __init__(self, mconfig: ModelParameters):
        self.mconfig = mconfig

    def save(self, path: Union[str, os.PathLike, BinaryIO, IO[bytes]], state_dict: dict):
        ndict = {
            "mconfig": dataclasses.asdict(self.mconfig),
            "model_state": state_dict
        }
        torch.save(ndict, path)

    def make_model(self, device):
        mconfig = self.mconfig
        if mconfig.arch == "calamari":
            return CalamariNetv2(mconfig.Channel, mconfig.Height, mconfig.Width, mconfig.seq_len, len(mconfig.voc),
                                 mconfig.hidden_units, mconfig.layers, mconfig.keep_prob, mconfig.seq_len, device,
                                 mconfig.leaky_relu)
        elif mconfig.arch == "crnn":
            return CRNN(mconfig.Channel, mconfig.Height, mconfig.Width, mconfig.seq_len, len(mconfig.voc),
                        mconfig.hidden_units, mconfig.layers, mconfig.keep_prob, mconfig.seq_len, device,
                        mconfig.leaky_relu, pre_lstm_dropout=mconfig.dropout, inter_lstm_dropout=mconfig.dropout)
        elif mconfig.arch == "stncrnn":
            return STNCRNN(mconfig.Channel, mconfig.Height, mconfig.Width, mconfig.seq_len, len(mconfig.voc),
                           mconfig.hidden_units, mconfig.layers, mconfig.keep_prob, mconfig.seq_len, device,
                           mconfig.leaky_relu)
        elif mconfig.arch == "trcrnn":
            return TrCRNN(mconfig.Channel, mconfig.Height, mconfig.Width, mconfig.seq_len, len(mconfig.voc),
                          mconfig.hidden_units, mconfig.layers, mconfig.keep_prob, mconfig.seq_len, device,
                          mconfig.leaky_relu)
        elif mconfig.arch == "symbolcrnn":
            return CRNNSymbol(mconfig.Channel, mconfig.Height, mconfig.Width, mconfig.seq_len, len(mconfig.voc),
                          mconfig.hidden_units, mconfig.layers, mconfig.keep_prob, mconfig.seq_len, device,
                          mconfig.leaky_relu)
        else:
            raise KeyError(f"invalid model specified: {mconfig.arch}")

    @staticmethod
    def load_model(path: Union[str, os.PathLike, BinaryIO, IO[bytes]], device) -> Tuple[
        'ModelConfiguration', nn.Module]:
        ndict = torch.load(path, map_location=lambda storage, loc: storage)
        mconfig = ModelParameters(**ndict["mconfig"])
        config = ModelConfiguration(mconfig)
        model = config.make_model(device)
        model.load_state_dict(ndict["model_state"], strict=False)
        model = model.to(device)
        # model = torch.nn.DataParallel(model).to(device)
        return config, model

def train_model(opt: TrainingOpts, seed: int = None) -> TrainingResult:
    if seed is not None:
        manualSeed = seed
    else:
        manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed:", )
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # turn on GPU for models:
    if opt.gpu == False:
        device = torch.device("cpu")
        print("CPU being used!")
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            device = torch.device("cuda")
            print("GPU being used!")
        else:
            device = torch.device("cpu")
            print("CPU being used!")

    print(opt)
    print("Creating Datasets")

    reader_train, reader_val = make_readers(opt)

    # build the vocabulary
    cmap, max_seq_len_ds = make_charmap_from_readers(
        [reader_train, reader_val])

    voc, char2id, id2char = cmap.charmap, cmap.char2id, cmap.id2char
    print("Training Vocabulary: ", voc)

    # set training parameters
    batch_size = opt.batch_size
    Height = opt.img_height
    Width = opt.img_width
    # Width = 640
    # Width = 1280
    feature_height = Height // 4
    feature_width = Width // 8
    Channel = opt.channel

    output_classes = len(voc)
    hidden_units = opt.hidden_units
    layers = opt.layers
    keep_prob = opt.keep_prob
    seq_len = opt.seq_len
    epochs = opt.epoch
    worker = opt.worker
    dataset_path = opt.dataset
    output_path = opt.output
    trained_model_path = opt.model

    if max_seq_len_ds > seq_len:
        seq_len = max_seq_len_ds + 2
        print(f"Warning: Changing Max Sequence Length to: {seq_len}")

    if not Path(output_path).exists():
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    # create model
    print("Create model......")
    mconfig = ModelConfiguration(ModelParameters(Height, Width, Channel, feature_height, feature_width,
                                 seq_len, opt.arch, hidden_units, layers, keep_prob, voc, char2id, id2char, False))

    if trained_model_path != '':
        print("Loading Trained Model Weights")
        mconfig, model = ModelConfiguration.load_model(
            trained_model_path, device)

        did_ajust, result = adjust_voc_model(voc, mconfig, model, device)
        mconfig, model = result.mconfig, result.model

        if did_ajust:
            print("Adjusted model encoding. More training is required")

        voc = mconfig.mconfig.voc
        id2char = mconfig.mconfig.id2char
        char2id = mconfig.mconfig.char2id

        # if torch.cuda.is_available() == True and opt.gpu == True:
        #    model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
        #    model = torch.nn.DataParallel(model).to(device)
        # else:
        #    model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
    else:
        model = mconfig.make_model(device)
        if torch.cuda.is_available() == True and opt.gpu == True:
            model = model.to(device)
        else:
            model = model.to(device)

    if torch.cuda.is_available() == True and opt.gpu == True:
        model = torch.nn.DataParallel(model).to(device)

    print("Loading Datasets")
    train_dataset = dataset.MySimpleDataset(
        reader_train, Height, Width, seq_len, char2id, augment=opt.augment)
    test_dataset = dataset.MySimpleDataset(
        reader_val, Height, Width, seq_len, char2id, augment=False)

    # make dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(worker))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(worker))

    print("Length of train dataset is: ", len(train_dataset))
    print("Length of test dataset is: ", len(test_dataset))
    print("Number of output classes is: ", train_dataset.output_classes)

    # make model output folder
    try:
        os.makedirs(output_path)
    except OSError:
        pass

    if opt.test_loaded_model and trained_model_path:
        print("Testing trained model")
        test_model(model, test_dataloader)

    #
    if opt.optimizer == "AdaBelief":
        optimizer = AdaBelief(model.parameters(),
                              print_change_log=False, lr=opt.lr)
    elif opt.optimizer == "Adam":
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "SGD":
        import torch.optim as optim
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    else:
        raise RuntimeError("Invalid optimizer specified")

    def lmbda(epoch): return 0.9**(epoch // 300) if epoch < 100 else 10**(-2)
    if False:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    else:
        scheduler = None

    num_batch = math.ceil(len(train_dataset) / batch_size)

    print("Beginning training")
    best_total_cer = float("inf")
#   model.module.rnn[-1].
    epochs_without_improvement = 0
    detailed_results = []
    for epoch in range(epochs):
        loss_avg = []
        progress = tqdm.tqdm(enumerate(train_dataloader),
                             total=len(train_dataloader))
        for i, data in progress:
            x = data[0]  # [batch_size, Channel, Height, Width]

            y = data[1]  # [batch_size, seq_len, output_classes]
            x, y = x.to(device), y.to(device)
            # show_images([x[0].cpu().permute(1,2,0).numpy()], ["Hello"])
            # print(x.shape, y.shape)
            optimizer.zero_grad()
            model = model.train()
            predict, _, _, _ = model(x, y)
            # print("Predict size: ", predict.size())
            # print("Target size: ", y.size())
            target = y.max(2)[1]  # [batch_size, seq_len]
            # print(target.cpu().numpy())
            try:
                target_firstpad = (target == char2id["PAD"]).cpu(
                ).numpy().argmax(axis=1).tolist()
                target_firstpad = [(x if x > 0 else seq_len)
                                   for x in target_firstpad]
                # print(target_firstpad)
                # sys.exit(1)
                # firstpads = [ int(x[0][0]) for x in target_firstpad.cpu().numpy() ]
                # print(firstpads)
                # target = target[:,:target_firstpad]
            except IndexError as e:
                pass
           # print(target.cpu().numpy())
            # print(target_firstpad)

            # show_images([np.transpose(x[0].cpu().numpy(),(1,2,0))],[decode_gt(target[0].cpu().numpy(),id2char)])
            # print("Prediction size is:", predict.shape)
            # [seq_len, batch_size, output_classes]
            predict_reshape = predict.permute(1, 0, 2)
            # print("Predict Reshaped: ", predict_reshape.size())
            # print("Target", target.size())
            predict_softmax = predict_reshape.log_softmax(2)
            input_lengths = tuple(
                [int(predict_reshape.size()[0])] * target.size()[0])
            # target_lengths = #tuple([target.size()[1]] * batch_size)
            target_lengths = target_firstpad
            # print("Inp Len: ", input_lengths)
            # print("Target Len: ", target_lengths)
            loss = nn.CTCLoss(blank=char2id["PAD"], reduction="sum", zero_infinity=True).forward(log_probs=predict_softmax,
                                                                                                 targets=target,
                                                                                                 input_lengths=input_lengths,
                                                                                                 target_lengths=target_lengths)
            loss_avg.append(float(loss))

            # from torch.nn.functional import ctc_loss
            # print("test")
            # print(ctc_loss(
            #    torch.as_tensor([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,1.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]),
            #    torch.as_tensor([2,1,1,2]),
            #    torch.as_tensor([6]),
            #    torch.as_tensor([4])))

            loss.backward()
            if opt.grad_clip:
                """
                total_norm = 0
                for p in model.parameters():
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Gradient norm: {total_norm:.3f}")
                """
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # prediction evaluation
            pred_choice = predict.max(2)[1]  # [batch_size, seq_len]

            progress.desc = f'[Epoch {epoch}: {i}/{num_batch}] train loss: {np.mean(loss_avg):.3f}'
            # print("predict prob:", predict[0][0])
            # print("predict words:", predict_words[0])
            # print("labeled words:", labeled_words[0])
        print("Epoch {} average loss: {:.5f} ".format(
            epoch, float(np.mean(loss_avg))))

        if scheduler is not None:
            scheduler.step()

        def print_val(gt, pred, id2char):
            # reverse char2id
            # id2char = {v:k for k,v in id2char.items()}
            gt_chars = []
            pred_chars = []
            alphabet = [0] * len(id2char)
            net_out = pred.cpu().numpy()
            print(net_out.shape)
            for k, v in id2char.items():
                alphabet[k] = v
            print("Alph: ", len(alphabet))
            print(alphabet)
            seq, _ = viterbi_search(net_out, alphabet)
            for x in gt:
                gt_chars.append(id2char[int(x)])
            print("GT: " + "".join(gt_chars))
            print("Pred ", seq)

        if epoch < 20:
            # continue
            pass
        # Validation
        print("Testing......")
        print_indices = set(random.choices(
            list(range(len(test_dataloader))), k=10))
        with torch.set_grad_enabled(False):
            CER_SUM = []
            total_chars: int = 0
            errors: int = 0
            model = model.eval()
            for i, data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                x = data[0]  # [batch_size, Channel, Height, Width]
                y = data[1]  # [batch_size, seq_len, output_classes]
                x, y = x.to(device), y.to(device)

                predict, _, _, _ = model(x, y)
                # prediction evaluation
                pred_choice = predict.max(2)[1]  # [batch_size, seq_len]
                pred_decoded = predict.argmax(dim=2)
                target = y.max(2)[1]  # [batch_size, seq_len]
                try:
                    target_firstpad = (target[0] == char2id["PAD"]).nonzero(
                        as_tuple=True)[0][0]
                    target = target[:, :target_firstpad]
                except IndexError as e:
                    pass
                # print(target_firstpad)
                # if i % 64 == 0:
                #    print_val(target[0], predict[0], id2char)
                decoded_gt = decode_gt(target[0], id2char)
                decoded_tgt = decode_ctc(predict[0], id2char)
                decoded_gt_eval = decoded_gt
                decoded_tgt_eval = decoded_tgt
                for c in opt.test_ignore_chars:
                    decoded_gt_eval = decoded_gt_eval.replace(c, '')
                    decoded_tgt_eval = decoded_tgt_eval.replace(c, '')
                edit_distance = editdistance.distance(
                    decoded_tgt_eval, decoded_gt_eval)
                CER = edit_distance / max(1, len(decoded_gt_eval))
                total_chars += max(1, len(decoded_gt_eval))
                errors += edit_distance
                CER_SUM.append(CER)
                if i in print_indices:
                    print(f"GT:   {decoded_gt}")
                    print(f"Pred: {decoded_tgt}")
                    print(f"  CER = {CER:.5f}")

            print("Epoch {} average CER: {:.5f}".format(
                epoch, float(np.mean(CER_SUM))))
            print("Epoch {} total CER: {:.5f}".format(
                epoch, errors / total_chars))
            with open(os.path.join(output_path, 'statistics.txt'), 'a') as f:
                f.write("TEST CER Epoch {} {}\n".format(
                    epoch, float(np.mean(CER_SUM))))
                f.write("Epoch {} total CER: {:.5f}\n".format(
                    epoch, errors / total_chars))
                f.write("Best CER (until now): {:.5f}\n".format(
                    best_total_cer))

            total_cer = errors / total_chars
            if total_cer < best_total_cer:
                best_total_cer = total_cer
                state_dict = None
                if torch.cuda.is_available() == True and opt.gpu == True:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                model_output_path = Path(output_path) / "model_best.pth"
                mconfig.save(model_output_path, state_dict)
                print(f"Saving model as {model_output_path}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement > opt.early_stopping:
                break

    print("Best total CER is:", best_total_cer)
    return TrainingResult(
        best_model_path=Path(output_path) / "model_best.pth",
        best_val_cer=best_total_cer
    )