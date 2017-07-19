# corr_spike_trains - This Python script implements algorithms based on [1]
# for generating correlated spike trains.
#
# [1] : Romain Brette, "Generation of Correlated Spike Trains", Neural
# Computation 21, 188-215, 2009.
#
# Copyright (C) 2016  Georgios Is. Detorakis (gdetor@protonmail.com)
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

import numpy as np
import matplotlib.pylab as plt
from corr_spike_trains import correlated_spikes


if __name__ == '__main__':
    if 1:
        n_proc = 5

        C = (np.ones((n_proc, n_proc)) +
             np.random.uniform(0, 1, (n_proc, n_proc)) * 5.0)
        np.fill_diagonal(C, [5, 6, 7, 8, 9])
        C = np.maximum(C, C.T)
        rates = np.array([5, 15, 4, 6, 7])

        cor_spk = correlated_spikes(C, rates, n_proc)
        spikes = cor_spk.cox_process(time=20000)
        cor_spk.raster_plot()
        spk = cor_spk.extract_pyNCS_list()
        plt.show()

    if 0:
        n_proc = 5
        P = np.random.randint(0, 2, (n_proc, n_proc))
        nu = np.random.random(n_proc) * 50

        C = (np.ones((n_proc, n_proc)) +
             np.random.uniform(0, 1, (n_proc, n_proc)) * 5.0)
        np.fill_diagonal(C, [5, 6, 7, 8, 9])
        C = np.maximum(C, C.T)
        rates = np.array([5, 15, 4, 6, 7])

        cor_spk = correlated_spikes(C, rates, n_proc)
        res = cor_spk.offline_mixture(P, nu, n_src=5, n_trg=5, time=500000)
