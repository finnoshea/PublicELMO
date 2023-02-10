import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from elm_finder_pkg import Elmo

from typing import List


class ELMCandidate:
    """
    """
    def __init__(self,
                 shot_number: int,
                 index: int,
                 times: np.array,
                 bes: np.array
                 ):
        self.shot_number = shot_number
        self.index = index
        self.times = times
        self.bes = bes
        self.is_elm = 0

    def populate_axis(self,
                      ax: plt.axis,
                      relative_time: bool = True
                      ) -> None:
        """ This assumes that the ELM peak is right in the middle of times. """
        ax.set_title(self.make_name())
        ax.plot(self.times, self.bes, 'b', alpha=0.5)
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('mean bes signal (V)')
        ax.set_ylim([-2, 10])
        dt = (self.times.shape[0] - 1) // 4
        time_ticks = self.times[np.arange(0, self.times.shape[0], dt)]
        t0 = 0.0  # keep absolute times
        if relative_time:
                t0 = self.times[self.times.shape[0] // 2]
        time_labels = ['{:4.1f}'.format(x - t0) for x in time_ticks]
        ax.set_xticks(time_ticks, labels=time_labels)
        ax.text(0.80, 0.93, '{:4.1f}'.format(t0), horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes,
                color='r')

    def package(self) -> dict:
        package = {}
        for key, value in self.__dict__.items():
            package[key] = value
        return package

    def label_as_elm(self):
        self.is_elm = 1

    def make_name(self) -> str:
        return '{:6d}-{:02d}'.format(self.shot_number, self.index)


class ELMFacebook:
    """
    """
    def __init__(self, shots_list: List[int]):
        self.shots_list = shots_list
        self.dt = 200  # 200 ms long window
        self.d_index = 1000  # how many microseconds to plot for each candidate
        self.candidates = []  # for holding ELMCandidates

    def __len__(self):
        return len(self.candidates)

    @staticmethod
    def _make_filename(shot_number: int,
                       direc: str = '/usr/src/app/elm_data'
                       ) -> str:
        fn = 'elm_data_{:d}.h5'.format(shot_number)
        return os.path.join(direc, fn)

    def pick_random_time(self, shot_file: str) -> float:
        with h5py.File(shot_file,'r') as hdf:
            t0 = hdf['BESFU']['times'][0]
            t1 = hdf['BESFU']['times'][-1] - self.dt
        return (t1 - t0) * np.random.rand() + t0

    def create_candidates(self):
        for shot in self.shots_list:
            fn = self._make_filename(shot)
            st = self.pick_random_time(fn)
            et = st + self.dt
            elmo = Elmo(fn, start_time=st, end_time=et, percentile=0.997)
            df = elmo.find_elms()
            elm_times = df['time'][df['peaks'] == True]
            for index, time_index in enumerate(elm_times.index):
                si = max(time_index - self.d_index, 0)
                ei = time_index + self.d_index + 1
                ec = ELMCandidate(shot_number=shot, index=index,
                                  times=elmo.data['time'][si:ei],
                                  bes=elmo.data['bes'][si:ei])
                ec.label_as_elm()
                # if elmo.data['time'][si:ei].size == 0:
                #     print('The following candidate did something strange:')
                #     print('shot: ', shot)
                #     print('index: ', index)
                #     print('si: ', si)
                #     print('ei: ', ei)
                #     print('times: ', elmo.data['time'][si:ei])
                #     print('bes: ', elmo.data['bes'][si:ei])
                self.candidates.append(ec)

    def package_candidates(self, filename: str = 'candidates.h5'):
        with h5py.File(filename, 'w') as hdf:
            for candidate in self.candidates:
                grp = hdf.create_group(candidate.make_name())
                for k, v in candidate.package().items():
                    grp.create_dataset(name=k, data=v)

    def plot_candidates(self, pdf_name: str = 'multipage_pdf.pdf'):
        with PdfPages(pdf_name) as pdf:
            for idx, candidate in enumerate(self.candidates):
                mod_idx = idx % 20
                if mod_idx == 0:
                    fig, axes = plt.subplots(5, 4, squeeze=True,
                                             figsize=(8.5, 11))
                    axes = axes.flatten()
                ax = axes[mod_idx]
                candidate.populate_axis(ax)
                if idx % 4 != 0:  # remove most of the y-labels
                    ax.yaxis.label = plt.text(0, 0.5, '')
                if mod_idx < 16:  # remove most of the x-labels
                    ax.xaxis.label = plt.text(0, 0.5, '')
                if mod_idx == 19 or idx + 1 == len(self.candidates):
                    # plt.title('Page One')
                    fig.set_tight_layout(True)
                    pdf.savefig(figure=fig)
                    plt.close(fig=fig)
            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'ELM Facebook'
            d['Author'] = "Finn O'Shea"


if __name__ == "__main__":
    import json
    from datetime import datetime
    t = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')

    # shots = [166576]
    with open('elm_shots.json', 'r') as df:
        shots = json.load(df)
    fb = ELMFacebook(shots_list=shots)
    t0 = time.time()
    fb.create_candidates()
    t1 = time.time()
    print('time to load {:d} ELMs from {:d} files: {:f}'.format(
        len(fb.candidates), len(shots), t1 - t0))
    fb.package_candidates('elm_candidates/candidates_' + t + '.h5')
    fb.plot_candidates('elm_candidates/candidates_' + t + '.pdf')
    t2 = time.time()
    print('time to plot {:d} ELMs from {:d} files: {:f}'.format(
        len(fb.candidates), len(shots), t2 - t1))




