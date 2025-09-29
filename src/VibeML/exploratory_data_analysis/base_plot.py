from pathlib import Path

import matplotlib.pyplot as plt

from ..data_processing.utils import str_to_path

class BasePlot:

    def __init__(self):
        self.fig, self.ax = plt.subplots(layout='constrained')
        self.summary_statistic = dict()

    def save(self, path: str | Path):
        path = str_to_path(path)
        self.fig.savefig(path)

    @property
    def describe(self) -> str:
        descriptions = []
        for statistic, summary in self.summary_statistic.items():
            descriptions.append(
                f"{statistic}: {summary}"
            )
        return "\n".join(descriptions)