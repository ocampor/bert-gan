from pathlib import Path
from typing import Tuple, Iterator, Dict

import pandas
import toolz


def split_data(x: str) -> Tuple[str, str, str]:
    sub_category, message = x.split(sep=" ", maxsplit=1)
    return toolz.first(sub_category.split(":")), sub_category, message


def parse_file(path: Path) -> Iterator[Tuple[str, str]]:
    with path.open("r") as fp:
        header = fp.readline()
        while line := fp.readline().strip():
            yield split_data(line)


def load_dataset(path: Path) -> pandas.DataFrame:
    records = parse_file(path)
    header = ["category", "sub-category", "message"]
    return pandas.DataFrame(records, columns=header)


def load_datasets(path: Path) -> Dict[str, pandas.DataFrame]:
    return {
        path.name: load_dataset(path)
        for path in path.glob("*.tsv")
    }
