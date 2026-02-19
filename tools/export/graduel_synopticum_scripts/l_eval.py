import json
import os
import re
import time

import tqdm

from database.file_formats.pcgts import SymbolType, ClefType, NoteType, AccidType, MusicSymbolPositionInStaff, \
    GraphicalConnectionType, MusicSymbol
from omr.experimenter.documents.b_evalluator import evaluate_symbols, evaluate_text


def parse_neural_output(output_line):
    # 1. Strip XML tags and line numbers
    content_match = re.search(r"<line>\d+\.\s*(.*?)</line>", output_line)
    if not content_match:
        return "", []

    content = content_match.group(1)

    # --- Extract Text ---
    # Remove all symbol blocks [...]
    text_only = re.sub(r"\[.*?\]", "", content)
    # Remove the initial start marker '*' if present
    text_only = text_only.replace("*", "")
    # Strip leading/trailing whitespace
    text_sequence = text_only.strip()

    # --- Extract Symbols ---
    # Find all content inside brackets [...]
    symbol_blocks = re.findall(r"\[(.*?)\]", content)

    music_symbols = []

    for block in symbol_blocks:
        # Regex to find tuples like (C_f|7|N)
        # Groups: 1=Type, 2=Pos, 3=GC
        raw_syms = re.findall(r"\(([^|]+)\|([^|]+)\|([^)]+)\)", block)

        for t_val, p_val, g_val in raw_syms:

            # Defaults
            s_type = SymbolType.NOTE
            n_type = None
            c_type = None
            a_type = None

            # Parse Type (e.g., C_f, N_0, A_n)
            parts = t_val.split('_')
            main_char = parts[0]
            sub_val = parts[1] if len(parts) > 1 else ""

            if main_char == 'C':
                s_type = SymbolType.CLEF
                # Map 'f' -> ClefType.F
                c_type = ClefType.F if sub_val == 'f' else ClefType.C
            elif main_char == 'N':
                s_type = SymbolType.NOTE
                # Map '0' -> NoteType.NORMAL (assuming int value match)
                try:
                    n_type = NoteType(int(sub_val))
                except:
                    n_type = NoteType.NORMAL
            elif main_char == 'A':
                s_type = SymbolType.ACCID
                if sub_val == 'flat':
                    a_type = AccidType.FLAT
                elif sub_val == 'sharp':
                    a_type = AccidType.SHARP
                else:
                    a_type = AccidType.NATURAL

            # Parse Position
            try:
                pos = MusicSymbolPositionInStaff(int(p_val))
            except ValueError:
                pos = MusicSymbolPositionInStaff.UNDEFINED

            # Parse Graphical Connection
            # "N" usually means None/Gaped for Clefs, otherwise it's an int string
            gc = GraphicalConnectionType.GAPED
            if g_val != "N" and g_val != "NONE":
                if s_type == SymbolType.NOTE:
                    try:
                        gc = GraphicalConnectionType(int(g_val))
                    except ValueError:
                        pass

            # Create Object
            ms = MusicSymbol(
                symbol_type=s_type,
                clef_type=c_type,
                note_type=n_type,
                accid_type=a_type,
                position_in_staff=pos,
                graphical_connection=gc
            )
            music_symbols.append(ms)

    return text_sequence, music_symbols


def load_ground_truth_map(jsonl_path):
    """
    Reads the JSONL file and creates a dictionary mapping
    UUIDs (extracted from image paths) to the Ground Truth text.
    """
    gt_map = {}

    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return {}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                entry = json.loads(line)

                # 1. Extract Image Filename to get UUID
                # Structure: messages -> user -> content -> image
                user_msg = next((m for m in entry["messages"] if m["role"] == "user"), None)
                if not user_msg: continue

                img_content = next((c for c in user_msg["content"] if c["type"] == "image"), None)
                if not img_content: continue

                # Extract "b3ffa4a6..." from "images/b3ffa4a6...jpg"
                image_path = img_content["image"]
                base_name = os.path.basename(image_path)  # e.g., b3ffa4a6...jpg
                uuid_key = os.path.splitext(base_name)[0]  # e.g., b3ffa4a6...

                # 2. Extract Ground Truth Text
                # Structure: messages -> assistant -> content -> text
                assist_msg = next((m for m in entry["messages"] if m["role"] == "assistant"), None)
                if not assist_msg: continue

                text_content = next((c for c in assist_msg["content"] if c["type"] == "text"), None)
                if not text_content: continue

                gt_text = text_content["text"]

                # Store in map
                gt_map[uuid_key] = gt_text

            except json.JSONDecodeError:
                print(f"Warning: Skipped invalid JSON on line {line_num + 1}")
            except Exception as e:
                print(f"Warning: Error processing line {line_num + 1}: {e}")

    return gt_map


def process_predictions(pred_folder, gt_map):
    """
    Iterates over prediction files, finds matches in gt_map,
    and yields pairs of (filename, prediction, ground_truth).
    """
    if not os.path.exists(pred_folder):
        print(f"Error: Folder {pred_folder} not found.")
        return

    # List all files ending in _pred_.txt
    files = [f for f in os.listdir(pred_folder) if f.endswith("_pred_.txt")]

    print(f"Found {len(files)} prediction files.")
    print(f"Found {len(gt_map)} ground truth entries.")
    print("-" * 50)

    matched_count = 0

    for filename in files:
        # Extract UUID from filename: "0cc4..._pred_.txt" -> "0cc4..."
        # We split by "_pred_" and take the first part
        uuid_key = filename.split("_pred_")[0]

        # Read the prediction text
        file_path = os.path.join(pred_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            prediction_text = f.read().strip()

        # Find matching Ground Truth
        if uuid_key in gt_map:
            gt_text = gt_map[uuid_key]
            matched_count += 1

            yield uuid_key, prediction_text, gt_text
        else:
            print(f"Warning: No Ground Truth found for {filename} (UUID: {uuid_key})")

    print("-" * 50)
    print(f"Processing complete. Matched {matched_count}/{len(files)} files.")



if __name__ == "__main__":
    raw_output = "<line>1. *[(C_f|7|N)] quos[(N_0|6|2)] pre[(N_0|8|2)]-ci[(N_0|9|2)]-nit[(N_0|6|2)] cla[(N_0|7|2)]-ri[(N_0|6|2)]-tas[(N_0|5|2)] iu[(N_0|7|2)]-bi[(N_0|5|2)]-lan[(N_0|4|2)]-de[(N_0|5|2)]-i[(N_0|5|2)]</line>"
    from xlwt import Workbook

    wb2 = Workbook()
    # Workbook is created
    wb = Workbook()
    # --- Main Execution ---
    JSONL_PATH = "/home/alexanderh/unsloth2/train_data.jsonl"
    PRED_FOLDER = "/home/alexanderh/unsloth2/images"

    # 1. Load Ground Truth into memory
    ground_truth_map = load_ground_truth_map(JSONL_PATH)

    # 2. Iterate and Process
    results = list(process_predictions(PRED_FOLDER, ground_truth_map))
    symbols_pred = []
    symbols_gt = []
    text =[]
    text_gt = []
    results_symbols_l = []
    results_text_l = []

    for uuid, pred, gt in tqdm.tqdm(results):
        #print(f"MATCH FOUND: {uuid}")
        #print(f"PRED: {pred[:50]}...")  # Printing first 50 chars for brevity
        #print(f"GT:   {gt[:50]}...")
        #print("")
        t_start = time.perf_counter()

        text_result, symbols_result = parse_neural_output(pred)
        text_result = text_result.replace("'", "").replace("-", "").lower()
        t_1 = time.perf_counter()

        print(f"{'parse_neural_output(pred)':<50} | {(t_1 - t_start) * 1000:>10.4f} ms")
        text_result_gt, symbols_result_gt = parse_neural_output(gt)
        text_result_gt = text_result_gt.replace("'", "").replace("-", "").lower()
        t_2 = time.perf_counter()

        print(f"{'parse_neural_output(pred)':<50} | {(t_2 - t_1) * 1000:>10.4f} ms")
        symbols_pred.append(symbols_result)
        symbols_gt.append(symbols_result_gt)
        text.append(text_result)
        text_gt.append(text_result_gt)
        print(symbols_result)
        print(symbols_result_gt)

        l = evaluate_symbols([symbols_result], [symbols_result_gt])
        results_symbols_l.append([("", ""), l, uuid])
        t_3 = time.perf_counter()

        print(f"{'parse_neural_output(pred)':<50} | {(t_3 - t_2) * 1000:>10.4f} ms")
        excel_text = evaluate_text([text_result], [text_result_gt])
        results_text_l.append([(text_result, text_result_gt), excel_text, uuid, ""])
        t_4 = time.perf_counter()

        print(f"{'parse_neural_output(pred)':<50} | {(t_4 - t_3) * 1000:>10.4f} ms")
        print("_____")
    sheet5 = wb.add_sheet('symbols Lines')

    ind = 3
    left = 3
    first = False
    for ind_d, d in enumerate(results_symbols_l):
        eval_data1, lines, docs_id = d
        p_str, gt_str = eval_data1
        if not first:
            for ind2, line in enumerate(lines[:-1]):
                sheet5.write(ind, 0, "Doc_id")
                sheet5.write(ind, 1, "p_str")
                sheet5.write(ind, 2, "gt_str")
                for ind1, cell in enumerate(line):
                    sheet5.write(ind, ind1 + left, str(cell))
                ind += 1
            first = True
        sheet5.write(ind, 0, str(docs_id))
        sheet5.write(ind, 1, str(p_str))
        sheet5.write(ind, 2, str(gt_str))

        for ind1, cell in enumerate(lines[-1]):
            sheet5.write(ind, ind1 + left, cell)
        ind += 1

    #for i in symbol_eval_data.doc_data:
    #    for f in i.eval_text:
    #        pred, gt = f.pred, f.gt
    #        excel_data = evaluate_text([pred], [gt])
    #        pred_str = pred
    #        gt_str = gt
    #        docs_instance_eval_data.append([(pred_str, gt_str), excel_data, i.doc_id, i.start_page_name])
    sheet6 = wb.add_sheet('text Lines')

    ind = 3
    left = 4
    first = False
    for ind_d, d in enumerate(results_text_l):
        eval_data1, lines, docs_id, start_page = d
        p_str, gt_str = eval_data1
        if not first:
            for ind2, line in enumerate(lines[:-1]):
                sheet6.write(ind, 0, "Doc_id")
                sheet6.write(ind, 1, "p_str")
                sheet6.write(ind, 2, "gt_str")
                sheet6.write(ind, 3, "start_page")

                for ind1, cell in enumerate(line):
                    sheet6.write(ind, ind1 + left, str(cell))
                ind += 1
            first = True
        sheet6.write(ind, 0, str(docs_id))
        sheet6.write(ind, 1, str(p_str))
        sheet6.write(ind, 2, str(gt_str))
        sheet6.write(ind, 3, str(start_page))

        for ind1, cell in enumerate(lines[-1]):
            sheet6.write(ind, ind1 + left, cell)
        ind += 1
        #print("\n--- Symbol Sequence ---")
        #for i, s in enumerate(symbols_result):
        #    print(f"{i + 1}. {s}")
    #text_result, symbols_result = parse_neural_output(raw_output)

    l = evaluate_symbols(symbols_pred, symbols_gt)
    sheet7 = wb.add_sheet('overall')

    ind = 3
    left = 3
    for i in l:
        for ind1, cell in enumerate(i):
            sheet7.write(ind, ind1 + left, cell)
        ind += 1

    ind = 8
    left = 3
    excel_text = evaluate_text(text, text_gt)
    for i in excel_text:
        for ind1, cell in enumerate(i):
            sheet7.write(ind, ind1 + left, cell)
        ind += 1

    wb.save(f"/tmp/evunsloth.xlsx")
