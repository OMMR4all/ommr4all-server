from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List
from prettytable import PrettyTable

from database import DatabaseBook
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--books", nargs="+", required=True)
    parser.add_argument("--ignore-page", nargs="+", default=[])
    parser.add_argument("--ignore-tools", nargs="+", default=[])

    args = parser.parse_args()

    all_stats = read_all_page_stats(args.books, args.ignore_page)

    print('Loaded {} pages from {} books'.format(len(all_stats), len(args.books)))

    stats = compute_avgs(all_stats, args.ignore_tools)

    pt = PrettyTable(['Tool', 'Time per page [min]', 'Total time [h]'])
    for n, t in stats.tool_timing_per_page.items():
        pt.add_row([n, t / 60, stats.total_tool_timing[n] / 3600])
    pt.add_row(["Total", stats.time_per_page * 60, stats.total_time])
    print(pt)

    pt = PrettyTable(['Action', 'Count'])
    for n, t in stats.total_actions.items():
        pt.add_row([n, t])
    print(pt)


