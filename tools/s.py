from omr.steps.text.hyphenation.hyphenator import CombinedHyphenator, HyphenDicts
from tools.simple_gregorianik_text_export import Lyric_info

if __name__ == "__main__":
    hyphen = CombinedHyphenator(lang=HyphenDicts.liturgical.get_internal_file_path(), left=1,
                                right=1)
    from glob import glob#
    import json
    files = glob("/home/alexanderh/Documents/datasets/rendered_files/*json")
    lines = []
    for i in files:
        with open(i, 'r', encoding='utf-8') as f:
            t = json.load(f)
            if "Precatus" in Lyric_info.from_dict(t).latine:

                print(t)

            lines.append(hyphen.apply_to_sentence(Lyric_info.from_dict(t).latine.replace(".", "")))
    with open("/home/alexanderh/Documents/datasets/allfileshyph.txt", 'w', encoding='utf-8') as f:
        for i in lines:
            f.write(i+"\n")
    c=0
    for i in lines:
        if len(i) > 256:
            c+=1


#with open(f"/tmp/files/{index}.json",  'w', encoding='utf-8') as f:
   #     json.dump(lyric_info.to_dict(), f, ensure_ascii=False, indent=4)