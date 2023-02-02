from typing import List

import numpy as np
from database.file_formats.pcgts import *

from database.file_formats.pcgts import MusicSymbol, SymbolPredictionConfidence, SymbolType, GraphicalConnectionType, \
    SymbolConfidence, ClefType, create_clef, create_accid, AccidType, Point
from omr.dataset import RegionLineMaskData
from omr.imageoperations.music_line_operations import SymbolLabel
from omr.steps.symboldetection.dataset import SymbolDetectionDataset
import cv2

def render_prediction_labels(labels, img=None):
    from shared.pcgtscanvas import PcGtsCanvas
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    if img is not None:
        out = np.stack((img,) * 3, axis=-1).astype(int)

    def draw(i, c):
        return np.kron((labels == i), c).reshape(out.shape).astype(int)

    for i, c in [
        (SymbolLabel.BACKGROUND, (255, 255, 255)),
        (SymbolLabel.NOTE_START, (255, 0, 0)),
        (SymbolLabel.NOTE_GAPPED, (255, 120, 120)),
        (SymbolLabel.NOTE_LOOPED, (120, 0, 0)),
        (SymbolLabel.CLEF_C, (120, 255, 120)),
        (SymbolLabel.CLEF_F, (0, 255, 0)),
        (SymbolLabel.ACCID_NATURAL, (0, 0, 255)),
        (SymbolLabel.ACCID_SHARP, (50, 50, 255)),
        (SymbolLabel.ACCID_FLAT, (0, 0, 120)),
    ]:
        c = PcGtsCanvas.color_for_music_symbol(i.to_music_symbol(), inverted=True, default_color=(255, 255, 255))
        if c != (0, 0, 0):
            out[:, :, 0] = np.where(labels == i, c[0], out[:, :, 0])
            out[:, :, 1] = np.where(labels == i, c[1], out[:, :, 1])
            out[:, :, 2] = np.where(labels == i, c[2], out[:, :, 2])

    # if img is not None:
    # out = (out.astype(float) * np.stack((img,) * 3, axis=-1) / 255).astype(np.uint8)

    return out.clip(0, 255).astype(np.uint8)
def extract_symbols(probs: np.ndarray, p: np.ndarray, m: RegionLineMaskData,
                    dataset: SymbolDetectionDataset, probability=0.5, clef=True, min_symbol_area=4, lookup= None) -> List[
    MusicSymbol]:
    # n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(((probs[:, :, 0] < 0.5) | (p > 0)).astype(np.uint8))
    p = (np.argmax(probs[:, :, 1:], axis=-1) + 1) * (probs[:, :, 0] < probability)
    #p = (np.argmax(probs, axis=-1))
    n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(p.astype(np.uint8))
    symbols = []
    sorted_labels = sorted(range(1, n_labels), key=lambda i: (centroids[i, 0], -centroids[i, 1]))
    centroids_canvas = np.zeros(p.shape, dtype=np.uint8)
    for i in sorted_labels:
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if a <= min_symbol_area and min_symbol_area > 0:
            continue
        y = stats[i, cv2.CC_STAT_TOP]
        x = stats[i, cv2.CC_STAT_LEFT]
        c = Point(x=centroids[i, 0], y=centroids[i, 1])
        coord = dataset.local_to_global_pos(c, m.operation.params)
        coord = m.operation.page.image_to_page_scale(coord, m.operation.scale_reference)
        # coord = coord.round().astype(int)

        # compute label this the label with the highest frequency of the connected component
        area = p[y:y + h, x:x + w] * (cc[y:y + h, x:x + w] == i)

        label = SymbolLabel(int(np.argmax([np.sum(area == v + 1) for v in range(len(SymbolLabel) - 1)])) + 1)
        centroids_canvas[int(np.round(c.y)), int(np.round(c.x))] = label

        # confidences
        indexes_of_cc = np.where(cc == i)
        labels_of_cc = p[indexes_of_cc]
        probs_of_cc = probs[indexes_of_cc]
        avg_prob_cc = np.mean(probs_of_cc, axis=0)
        symbol_pred = SymbolPredictionConfidence(*avg_prob_cc.tolist())
        if label == SymbolLabel.NOTE_START:
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord)
            # confidence_position_in_staff = m.operation.music_line.compute_confidence_position_in_staff(coord)
            # print(confidence_position_in_staff)
            symbols.append(MusicSymbol(
                symbol_type=SymbolType.NOTE,
                coord=coord,
                position_in_staff=position_in_staff,
                graphical_connection=GraphicalConnectionType.NEUME_START,
                confidence=SymbolConfidence(symbol_pred, None)
            ))
        elif label == SymbolLabel.NOTE_GAPPED:
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord)

            symbols.append(MusicSymbol(
                symbol_type=SymbolType.NOTE,
                coord=coord,
                position_in_staff=position_in_staff,
                graphical_connection=GraphicalConnectionType.GAPED,
                confidence=SymbolConfidence(symbol_pred, None)

            ))
        elif label == SymbolLabel.NOTE_LOOPED:
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord)

            symbols.append(MusicSymbol(
                symbol_type=SymbolType.NOTE,
                coord=coord,
                position_in_staff=position_in_staff,
                graphical_connection=GraphicalConnectionType.LOOPED,
                confidence=SymbolConfidence(symbol_pred, None)

            ))
        elif label == SymbolLabel.CLEF_C:
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord, clef=clef)
            coord_updated = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(coord.x,
                                                                                                  position_in_staff)
            symbols.append(create_clef(ClefType.C, coord=coord_updated, position_in_staff=position_in_staff,
                                       confidence=SymbolConfidence(symbol_pred, None)))
        elif label == SymbolLabel.CLEF_F:
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord, clef=clef)
            coord_updated = m.operation.music_line.staff_lines.compute_coord_by_position_in_staff(coord.x,
                                                                                                  position_in_staff)
            symbols.append(create_clef(ClefType.F, coord=coord_updated, position_in_staff=position_in_staff,
                                       confidence=SymbolConfidence(symbol_pred, None)))
        elif label == SymbolLabel.ACCID_FLAT:
            symbols.append(
                create_accid(AccidType.FLAT, coord=coord, confidence=SymbolConfidence(symbol_pred, None)))
        elif label == SymbolLabel.ACCID_SHARP:
            symbols.append(
                create_accid(AccidType.SHARP, coord=coord, confidence=SymbolConfidence(symbol_pred, None)))
        elif label == SymbolLabel.ACCID_NATURAL:
            symbols.append(
                create_accid(AccidType.NATURAL, coord=coord, confidence=SymbolConfidence(symbol_pred, None)))
        else:
            raise Exception("Unknown label {} during decoding".format(label))
        pass

    if False:
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(5, 1, sharex='all', sharey='all')
        ax[0].imshow(p)
        ax[1].imshow(render_prediction_labels(centroids_canvas, m.region))
        labels = render_prediction_labels(p, 255 - m.region)
        ax[2].imshow(labels)
        ax[3].imshow(m.region, cmap='gray_r')
        ax[4].imshow(cc, cmap='gist_ncar_r')
        plt.show()
    line = Line(symbols=symbols)
    if lookup:
        line.update_sequence_confidence(lookup)
    symbols = line.symbols

    return symbols


    def extract_symbols2(probs: np.ndarray, p: np.ndarray, m: RegionLineMaskData,
                         dataset: SymbolDetectionDataset) -> List[MusicSymbol]:
        # n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(((probs[:, :, 0] < 0.5) | (p > 0)).astype(np.uint8))
        p = (np.argmax(probs, axis=-1))
        n_labels, cc, stats, centroids = cv2.connectedComponentsWithStats(p.astype(np.uint8))
        symbols = []
        sorted_labels = sorted(range(1, n_labels), key=lambda i: (centroids[i, 0], -centroids[i, 1]))
        centroids_canvas = np.zeros(p.shape, dtype=np.uint8)
        for i in sorted_labels:
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            # if a <= 4:
            #    continue
            y = stats[i, cv2.CC_STAT_TOP]
            x = stats[i, cv2.CC_STAT_LEFT]
            c = Point(x=centroids[i, 0], y=centroids[i, 1])
            coord = dataset.local_to_global_pos(c, m.operation.params)
            coord = m.operation.page.image_to_page_scale(coord, m.operation.scale_reference)
            # coord = coord.round().astype(int)

            # compute label this the label with the hightest frequency of the connected component
            area = p[y:y + h, x:x + w] * (cc[y:y + h, x:x + w] == i)
            label = SymbolLabel(int(np.argmax([np.sum(area == v + 1) for v in range(len(SymbolLabel) - 1)])) + 1)
            centroids_canvas[int(np.round(c.y)), int(np.round(c.x))] = label
            position_in_staff = m.operation.music_line.compute_position_in_staff(coord)
            if label == SymbolLabel.NOTE_START:
                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.NEUME_START,
                ))
            elif label == SymbolLabel.NOTE_GAPPED:
                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.GAPED,
                ))
            elif label == SymbolLabel.NOTE_LOOPED:
                symbols.append(MusicSymbol(
                    symbol_type=SymbolType.NOTE,
                    coord=coord,
                    position_in_staff=position_in_staff,
                    graphical_connection=GraphicalConnectionType.LOOPED,
                ))
            elif label == SymbolLabel.CLEF_C:
                symbols.append(create_clef(ClefType.C, coord=coord, position_in_staff=position_in_staff))
            elif label == SymbolLabel.CLEF_F:
                symbols.append(create_clef(ClefType.F, coord=coord, position_in_staff=position_in_staff))
            elif label == SymbolLabel.ACCID_FLAT:
                symbols.append(create_accid(AccidType.FLAT, coord=coord))
            elif label == SymbolLabel.ACCID_SHARP:
                symbols.append(create_accid(AccidType.SHARP, coord=coord))
            elif label == SymbolLabel.ACCID_NATURAL:
                symbols.append(create_accid(AccidType.NATURAL, coord=coord))
            else:
                raise Exception("Unknown label {} during decoding".format(label))

        if False:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(6, 1, sharex='all', sharey='all')
            ax[0].imshow(p)
            ax[1].imshow(p)
            ax[2].imshow(render_prediction_labels(centroids_canvas, m.region))
            labels = render_prediction_labels(p, 255 - m.region)
            ax[3].imshow(labels)
            ax[4].imshow(m.region, cmap='gray_r')
            ax[5].imshow(cc, cmap='gist_ncar_r')
            plt.show()

        return symbols

    def extract_symbols_nms(probs: np.ndarray, p: np.ndarray, m: RegionLineMaskData,
                            dataset: SymbolDetectionDataset, threshold=0.35, symbol_area=10) -> List[MusicSymbol]:
        symbols = []
        shape = np.shape(probs)
        centroids_canvas = np.zeros(p.shape, dtype=np.uint8)

        for i in range(1, shape[-1]):
            class_probs = probs[:, :, i]
            while True:
                maximum = np.where(class_probs == np.amax(class_probs))
                if class_probs[maximum][0] < threshold:
                    break
                c = Point(x=maximum[1][0], y=maximum[0][0])

                class_probs[max(0, c.y - symbol_area): min(c.y + symbol_area, shape[0]),
                max(0, c.x - symbol_area): min(c.x + symbol_area, shape[1])] \
                    = 0

                coord = dataset.local_to_global_pos(c, m.operation.params)
                coord = m.operation.page.image_to_page_scale(coord, m.operation.scale_reference)
                label = SymbolLabel(i)
                centroids_canvas[int(np.round(c.y)), int(np.round(c.x))] = label
                position_in_staff = m.operation.music_line.compute_position_in_staff(coord)
                if label == SymbolLabel.NOTE_START:
                    symbols.append(MusicSymbol(
                        symbol_type=SymbolType.NOTE,
                        coord=coord,
                        position_in_staff=position_in_staff,
                        graphical_connection=GraphicalConnectionType.NEUME_START,
                    ))
                elif label == SymbolLabel.NOTE_GAPPED:
                    symbols.append(MusicSymbol(
                        symbol_type=SymbolType.NOTE,
                        coord=coord,
                        position_in_staff=position_in_staff,
                        graphical_connection=GraphicalConnectionType.GAPED,
                    ))
                elif label == SymbolLabel.NOTE_LOOPED:
                    symbols.append(MusicSymbol(
                        symbol_type=SymbolType.NOTE,
                        coord=coord,
                        position_in_staff=position_in_staff,
                        graphical_connection=GraphicalConnectionType.LOOPED,
                    ))
                elif label == SymbolLabel.CLEF_C:
                    symbols.append(create_clef(ClefType.C, coord=coord, position_in_staff=position_in_staff))
                elif label == SymbolLabel.CLEF_F:
                    symbols.append(create_clef(ClefType.F, coord=coord, position_in_staff=position_in_staff))
                elif label == SymbolLabel.ACCID_FLAT:
                    symbols.append(create_accid(AccidType.FLAT, coord=coord))
                elif label == SymbolLabel.ACCID_SHARP:
                    symbols.append(create_accid(AccidType.SHARP, coord=coord))
                elif label == SymbolLabel.ACCID_NATURAL:
                    symbols.append(create_accid(AccidType.NATURAL, coord=coord))
                else:
                    raise Exception("Unknown label {} during decoding".format(label))

        if True:
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(4, 1, sharex='all', sharey='all')
            ax[0].imshow(p)
            ax[1].imshow(render_prediction_labels(centroids_canvas, m.region))
            labels = render_prediction_labels(p, 255 - m.region)
            ax[2].imshow(labels)
            ax[3].imshow(m.mask, cmap='gray_r')
            # ax[4].imshow(cc, cmap='gist_ncar_r')
            import datetime
            t = datetime.datetime.now()
            plt.savefig(str(t) + '.png', dpi=180)

        return symbols