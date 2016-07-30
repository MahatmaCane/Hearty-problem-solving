#### Copyright Joel Dyer, 29/07/2016 ####

import numpy as np
import random

class Substrate:

	def __init__(self, tissue_shape=(5, 5), nu=0.5, delta=0.2, p=0.4, refractory_period=5):

		assert isinstance(tissue_shape, tuple), 'Expected tuple, got {}'.type(tissue_shape)
		self.shape = tissue_shape
		self._nu = nu
		self._delta = delta
		self._p = p
		self._refractory_period = refractory_period

		self.determine_locations_of_lateral_couplings()
		self.determine_locations_of_defective_cells()

		self.counts_until_relaxed = np.zeros(self.shape)
		self.wavefront = np.zeros(self.shape)
		self.pulse()
		
		self.find_excitable_cells()

	def determine_locations_of_lateral_couplings(self):

		random = np.random.random(self.shape)
		self.lateral_couplings = random <= self._nu

	def determine_locations_of_defective_cells(self):

		random = np.random.random(self.shape)
		self._defective_cells = random <= self._delta

	def find_excitable_cells(self):

		self.excitable_cells = self.counts_until_relaxed == 0

	def pulse(self):

		self.find_excitable_cells()
		ones = self.find_cells_which_will_fire()
		self.wavefront[:, 0] += ones[:, 0] * self.excitable_cells[:, 0]
		self.counts_until_relaxed[:,0] += ones[:, 0] * self.excitable_cells[:, 0] * self._refractory_period

	def update_counts_until_relaxed(self):

		refractory_cells = self.counts_until_relaxed > 0
		self.counts_until_relaxed[refractory_cells] -= 1
		self.counts_until_relaxed += self.wavefront * self._refractory_period 

	def evolve_wavefront(self):

		should_excite = np.zeros(self.shape, dtype=np.float)
		self.find_excitable_cells()
		should_excite[:, :-1] += self.wavefront[:, 1:] * self.excitable_cells[:, :-1]
		should_excite[:, 1:] += self.wavefront[:, :-1] * self.excitable_cells[:, 1:]
		should_excite += np.roll(self.wavefront, -1, axis=0) * self.lateral_couplings * self.excitable_cells
		should_excite += np.roll(self.wavefront, 1, axis=0) * self.lateral_couplings * self.excitable_cells
		should_excite[should_excite > 1] = 1

		ones = self.find_cells_which_will_fire()

		self.wavefront = ones * should_excite

	def find_cells_which_will_fire(self):

		ones = np.ones(self.shape)
		ones[self._defective_cells] = np.random.random(np.sum(self._defective_cells))
		ones[ones <= self._p] = 0
		ones[ones > self._p] = 1
		return ones
