from thirdparty.calamari.calamari_ocr.ocr.predictor import Predictor, MultiPredictor
from thirdparty.calamari.calamari_ocr.ocr.voting import voter_from_proto
from thirdparty.calamari.calamari_ocr.proto import VoterParams, Predictions
from typing import List
from omr.datatypes import PcGts
from omr.dataset.pcgtsdataset import PcGtsDataset
import main.book as book


class OMRPredictor:
    def __init__(self, checkpoints: List[str], ):
        self.predictor = MultiPredictor(checkpoints)
        self.height = self.predictor.predictors[0].network_params.features
        voter_params = VoterParams()
        voter_params.type = VoterParams.CONFIDENCE_VOTER_DEFAULT_CTC
        self.voter = voter_from_proto(voter_params)

    def predict(self, pcgts_files: List[PcGts]):
        dataset = PcGtsDataset(pcgts_files, gt_required=False, height=self.height)
        for r, sample in self.predictor.predict_dataset(dataset.to_calamari_dataset()):
            prediction = self.voter.vote_prediction_result(r)
            prediction.id = "voted"
            print(prediction.sentence, [(p.global_start, p.chars[0].char) for p in prediction.positions])
            yield r


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from omr.dewarping.dummy_dewarper import dewarp
    b = book.Book('Graduel')
    pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[:3]]
    val_pcgts = [PcGts.from_file(p.file('pcgts')) for p in b.pages()[3:4]]
    page = book.Book('Graduel').page('Graduel_de_leglise_de_Nevers_023')
    # pcgts = PcGts.from_file(page.file('pcgts'))
    pred = OMRPredictor([b.local_path('omr_models/omr_best_.ckpt')])
    ps = list(pred.predict(val_pcgts))
