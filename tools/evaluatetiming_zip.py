import os
import zipfile
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List
from prettytable import PrettyTable

from database import DatabaseBook, DatabasePage
from database.file_formats.performance.statistics import Statistics


def read_all_page_stats(books: List[str], ignore_page: List[str] = []):
    all_stats = []
    for book in books:
        book = DatabaseBook(book)
        for page in book.pages():
            if any([s in page.page for s in ignore_page]):
                continue
            all_stats.append(page.page_statistics())

    return all_stats


@dataclass
class Stats:
    total_pages: int
    total_tool_timing: dict
    tool_timing_per_page: dict
    total_time: float
    time_per_page: float

    total_actions: dict


def compute_avgs(stats: List[Statistics], ignore_tools):
    # timing stats
    total_tool_timing = {}
    for s in stats:
        for n, t in s.tool_timing.items():
            if any([it in n.lower() for it in ignore_tools]):
                continue

            total_tool_timing[n] = total_tool_timing.get(n, 0) + t / 1000  # ms -> s

    total_time = sum([t for n, t in total_tool_timing.items()]) / 3600  # s -> h

    # action stats
    total_actions = {}
    for s in stats:
        for n, t in s.actions.items():
            total_actions[n] = total_actions.get(n, 0) + t

    out = Stats(
        total_pages=len(stats),
        total_tool_timing=total_tool_timing,
        tool_timing_per_page={n: k / len(stats) for n, k in total_tool_timing.items()},
        total_time=total_time,
        time_per_page=total_time / len(stats),
        total_actions=total_actions,
    )

    return out


def load_file_from_file_and_zip(book: str):

    def statistics_to_worksheet(worksheet, stat: Statistics, ind = 0, name = "", dbpage: DatabasePage=None):
        symbols = [line.symbols for line in dbpage.pcgts().page.all_music_lines()]
        syllabels = [line.sentence.syllables for line in dbpage.pcgts().page.all_text_lines()]
        symbols = sum([len(s) for s in symbols])
        syllabels = sum([len(s) for s in syllabels])
        total_tool_timing = {}
        if ind ==1:
            worksheet.write(0, 1, "Date")
            worksheet.write(0, 2, "Total Time")
            worksheet.write(0, 3, "Total Actions")
            worksheet.write(0, 4, "Total: Symbols")
            worksheet.write(0, 5, str(symbols))
            worksheet.write(0, 6, "Total: Syllabels")
            worksheet.write(0, 7, str(syllabels))

        for n, t in stat.tool_timing.items():
            total_tool_timing[n] = total_tool_timing.get(n, 0) + t / 1000  # ms -> s
        total_time = sum([t for n, t in total_tool_timing.items()]) / 3600 * 60  # s -> h
        total_actions = {}
        for n, t in stat.actions.items():
            total_actions[n] = total_actions.get(n, 0) + t
        worksheet.write(ind+1, 1, name)
        worksheet.write(ind+1, 2, total_time)
        total_actinos = sum([t for n, t in total_actions.items()])

        worksheet.write(ind+1, 3, total_actinos)
        i = 4
        for n, t in stat.tool_timing.items():
            worksheet.write(ind+1, i, str(n))
            worksheet.write(ind+1, i+1, t / 3600 * 60 / 1000)
            i += 2
        i += 2

        for n, t in stat.actions.items():
            print(n)
            worksheet.write(ind+1, i, str(n))
            worksheet.write(ind+1, i+1, t)
            i += 2

    import xlsxwriter
    workbook = xlsxwriter.Workbook('/tmp/Statistic_Evaluation.xlsx')
    book = DatabaseBook(book)
    for page in book.pages():
        worksheet = workbook.add_worksheet(page.page)

        zip_file = page.file('statistics_backup').local_path()

        t = 0
        print(zip_file)
        if os.path.exists(zip_file):
            zip = zipfile.ZipFile(zip_file)
            for ind, t in enumerate(zip.namelist()):
                with zip.open(t) as f:
                    stats = Statistics.from_file(f)
                    statistics_to_worksheet(worksheet, stats, (ind + 1), t, page)
                    print(stats.to_json())
                    t = ind + 1
        stats = page.page_statistics()

        statistics_to_worksheet(worksheet, stats, t, "newest", page)

    pass
    workbook.close()

if __name__ == '__main__':
    book = "mul_2_23_gt_ina"

    all_stats = load_file_from_file_and_zip(book)


