import json
import os

import edlib

from database import DatabaseBook
from database.database_book_documents import DatabaseBookDocuments
from database.start_up.load_text_variants_in_memory import lyrics, syllable_dictionary
from ommr4all import settings
from omr.dataset.dataset import LyricsNormalizationProcessor, LyricsNormalizationParams, LyricsNormalization, logger
from tools.simple_gregorianik_text_export import Lyric_info, Lyrics

if __name__ == "__main__":
    book = DatabaseBook('mul_2_rsync_gt')
    documents = DatabaseBookDocuments().load(book)
    docs = documents.database_documents.documents
    text_normalizer = LyricsNormalizationProcessor(LyricsNormalizationParams(LyricsNormalization.WORDS))
    path = os.path.join(settings.BASE_DIR, 'internal_storage', 'resources', 'lyrics_collection',
                        'lyrics_by_sources.json')
    with open(path) as f:
        json1 = json.load(f)
        lyrics = Lyrics.from_dict(json1)
    logger.info("Successfully imported Lyrics database into memory")
    for i in documents.database_documents.documents:
        if True or i.document_meta_infos.genre == "":
            text = i.get_text_of_document(book=book)
            text = text_normalizer.apply(text)
            lowest_ed = 999999
            lowest_text = ""
            # ed = edlib.align("abc", "abc", mode="SHW", k=lowest_ed)
            lyric_info: Lyric_info = None
            for b in lyrics.lyrics:
                b: Lyric_info = b
                text2 = text_normalizer.apply(b.latine)
                ed = edlib.align(text.replace(" ", ""), text2.replace(" ", ""), mode="SHW", k=lowest_ed)
                if 0 < ed["editDistance"] < lowest_ed:
                    lowest_ed = ed["editDistance"]
                    lowest_text = text2
                    lyric_info = b
                elif text.replace(" ", "") == text2.replace(" ", ""):
                    lowest_ed = 0
                    lowest_text = text2
                    lyric_info = b
            if lyric_info:
                i.document_meta_infos.genre = lyric_info.genre
                i.document_meta_infos.url = lyric_info.url
                i.document_meta_infos.festum = lyric_info.meta_info
                festum = ""
                if lyric_info.meta_infos_extended:
                    if len(lyric_info.meta_infos_extended) > 0:
                        if lyric_info.meta_infos_extended[0].festum is not None:
                            festum += lyric_info.meta_infos_extended[0].festum
                        if lyric_info.meta_infos_extended[0].dies is not None:
                            festum += lyric_info.meta_infos_extended[0].dies
                i.document_meta_infos.festum = festum
                i.document_meta_infos.initium = lyric_info.initium
                i.document_meta_infos.dataset_source = lyric_info.dataset_source
                i.document_meta_infos.cantus_id = lyric_info.cantus_id
                i.textinitium=lyric_info.initium
                if lyric_info.initium == "" or lyric_info.initium is None:
                    i.document_meta_infos.initium = " ".join(text.split(" ")[:5])
                    i.textinitium = " ".join(text.split(" ")[:5])
                    if text.split(" ")[:5] is None:
                        print(text.split(" ")[:5])
                        print(text)
    documents.to_file(book)

