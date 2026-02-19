import docx
import re
import os

from database import DatabaseBook
from database.file_formats.pcgts.page import Sentence
from tools.export.asissi.align import solve_alignment


def get_text_from_docx(filename):
    try:
        doc = docx.Document(filename)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        return None


def parse_medieval_lyrics_v3(text):
    extracted_chants = []
    lines = text.strip().split('\n')

    current_chant = {
        "page": "Unbekannt",
        "lyrics": []
    }

    capturing_lyrics = False


    page_pattern = re.compile(r"\(\|*\s*f\.\s*(\d+[a-z]?)\)")

    stanza_start_pattern = re.compile(r"^(\d+[a-z]?)\s+")
    title_stop_pattern = re.compile(r"(;\s*AH\s+\d+)|(<[A-Z\s]{3,}>)")

    for line in lines:
        line = line.strip()
        if not line: continue

        page_match = page_pattern.search(line)
        if page_match:
            if capturing_lyrics and len(current_chant["lyrics"]) > 0:
                extracted_chants.append(current_chant)
                capturing_lyrics = False

            current_chant = {
                "page": f"f. {page_match.group(1)}",
                "lyrics": []
            }
            continue

        stanza_match = stanza_start_pattern.search(line)
        if stanza_match:
            capturing_lyrics = True
            clean_line = stanza_start_pattern.sub("", line)
            current_chant["lyrics"].append(clean_line)
            continue

        if capturing_lyrics:
            if title_stop_pattern.search(line):
                extracted_chants.append(current_chant)
                capturing_lyrics = False
                current_chant = {
                    "page": current_chant["page"],
                    "lyrics": []
                }
                continue
            else:
                current_chant["lyrics"].append(line)

    if capturing_lyrics and current_chant["lyrics"]:
        extracted_chants.append(current_chant)

    return extracted_chants


from collections import defaultdict



def organize_by_page(chants_list):
    """
    Reorganizes chant-based data into page-based data.
    Only creates new lines when '|' is encountered.
    Handles '||' as page transitions.
    """
    pages_content = defaultdict(lambda: [""])
    page_ref_pattern = re.compile(r"f\.\s*(\d+[a-z]?)")

    for chant in chants_list:
        current_page = chant['page']
        print(current_page)
        if pages_content[current_page][-1] != "":
            pages_content[current_page].append("")
        for line in chant['lyrics']:
            if " f. 76" in line:
                print(f"Found line with page ref: {line}")
            if "||" in line:
                segments = line.split("||")
                add_text_to_page(segments[0], pages_content[current_page])
                last_segment = segments[-1]
                page_match = page_ref_pattern.search(last_segment)

                if page_match:
                    current_page = f"f. {page_match.group(1)}"
                    if len(segments) > 2:
                        for seg in segments[1:-1]:
                            add_text_to_page(seg, pages_content[current_page])
                else:
                    print(f"No page number found in segment: {last_segment}, chant: {chant['page']}")
                    for seg in segments[1:]:
                        add_text_to_page(seg, pages_content[current_page])

            else:
                add_text_to_page(line, pages_content[current_page])

    final_pages = {}
    for page, lines in pages_content.items():
        print(page)
        clean_lines = [l for l in lines if l.strip()]
        if clean_lines:
            print(f"{page} added")

            final_pages[page] = clean_lines

    return final_pages


def add_text_to_page(text, page_lines_list):
    """
    Appends text to the current line of the page.
    Starts a new line ONLY if a '|' is found.
    """
    if not text:
        return

    parts = text.split('|')
    first_part = parts[0].strip()
    if first_part:
        if page_lines_list[-1]:
            page_lines_list[-1] += " " + first_part
        else:
            page_lines_list[-1] = first_part

    for part in parts[1:]:
        page_lines_list.append(part.strip())


def get_page_sort_key(page_label):
    """
    Sort key for page labels like "f. 56", "f. 5v", "f. 10".
    Returns a tuple (number, suffix) for proper numerical sorting.
    """
    match = re.search(r"(\d+)([a-z]*)", page_label)
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        return (num, suffix)
    return (0, "")

def remove_angle_brackets(text):
    """
    Removes everything between < and > (inclusive) from the string.
    """

    return re.sub(r'<[^>]*>', '', text)





if __name__ == "__main__":
    input_filename = "/tmp/Ass 695 Texte_AP.docx"
    output_filename = "/tmp/exported_lyrics.txt"


    if os.path.exists(input_filename):
        print(f"Verarbeite {input_filename}...")
        raw_text = get_text_from_docx(input_filename)

        if raw_text:
            results = parse_medieval_lyrics_v3(raw_text)

            with open(output_filename, "w", encoding="utf-8") as f:
                for i, chant in enumerate(results, 1):
                    f.write(f"PAGE: {chant['page']}\n")
                    f.write("-" * 20 + "\n")
                    f.write("\n".join(chant['lyrics']))
                    f.write("\n\n" + "=" * 30 + "\n\n")

            print(f"{len(results)} Liedtexte extrahiert.")
            print("Organizing lyrics by page...")

            pages_data = organize_by_page(results)

            output_filename = "lyrics_by_page.txt"


            with open(output_filename, "w", encoding="utf-8") as f:
                for page_num in pages_data.keys():
                    lines = pages_data[page_num]

                    header = f"=== CONTENT OF PAGE {page_num} ==="
                    f.write(header + "\n")

                    for line in lines:
                        f.write(line + "\n")

                    f.write("\n\n")

            print(f"Exported organized pages to {output_filename}")
            book = DatabaseBook("Ass695t")
            pages = book.pages()
            skip = True
            index = 0
            pages_data_lyric_list = [i for i in pages_data.values()]
            pages_data_lyric_list_keys = [i for i in pages_data.keys()]
            print("Pages data lyric list keys:")

            show = False
            for page in pages:
                if page.page == "folio_0056":
                    skip = False
                if skip:
                    continue

                v_page = False
                page_l = []
                try:
                    if page.page[-1] == 'v':
                        v_page = True
                        page_f = str(int(page.page[6:-1]))
                        print("key:", f"f. {page_f}v")
                        page_l = pages_data[f"f. {page_f}v"]
                    else:
                        page_f = str(int(page.page[6:]))

                        page_l = pages_data[f"f. {page_f}"]
                        print("key:", f"f. {page_f}")
                except:
                    print(f"Page {page.page} not found in lyrics data.")
                    index += 1
                    continue

                print("----------------")
                lines = page.pcgts().page.all_text_lines()
                if len(lines) == 0:
                    index += 1
                    continue
                lb = [l.text() for l in lines]
                la = [remove_angle_brackets(te) for te in page_l
                      ]
                la2 = [te for te in page_l
                       ]
                if True:
                    print("List la")
                    for i, l in enumerate(la):
                        print(f"{i:<5} {l}")
                    print("List lb")
                    for i, l in enumerate(lb):
                        print(f"{i:<5} {l}")
                    print("Alignment results:")
                results1 = solve_alignment(la, lb)
                sentences = {}

                for i, b_idx in enumerate(results1):
                    print(f"{i: <15} {la[i][:35]: <35} {b_idx: <5} {lb[b_idx]}")
                    if b_idx not in sentences:
                        sentences[b_idx] = la2[i]
                    else:
                        sentences[b_idx] += " " + la2[i]
                for i in range(len(lines)):
                    if i not in sentences:
                        sentences[i] = ""
                for val, key in sentences.items():
                    pass
                    print(f"Setting line {val} to sentence: {key}")
                    lines[val].sentence =  Sentence.from_string(key)
                page.pcgts().to_file(page.file('pcgts').local_path())
                index += 1
                if page.page == "0284":
                    show = False
                if show:
                    input("Press Enter to continue...")

    else:
        print("Datei nicht gefunden.")


