#### Copyright Joel Dyer, 29/07/2016 ####

import numpy as np

class Myocardium:

    def __init__(self, tissue_shape=(5, 5), d=0.01, e=0.05, refractory_period=50):

        self.shape = tissue_shape
        self._delta = d
        self._epsilon = e
        self._refractory_period = refractory_period

        self.__determine_random_lateral_coupling_array()
        self.__determine_locations_of_defective_cells()

        self.couplings_set = False

        self.counts_until_relaxed = np.zeros(self.shape, dtype=np.float64)
        self.wavefront = np.zeros(self.shape, dtype=np.bool)
        self.permanently_unexcitable = np.zeros(tissue_shape)

    def __determine_random_lateral_coupling_array(self):

        self.__random_lateral = np.random.random(self.shape)

    def __determine_locations_of_defective_cells(self):

        random = np.random.random(self.shape)
        self._defective_cells = random <= self._delta

    def __find_cells_to_excite(self):

        """Determines cells which are to excite in response to stimulus in the
        next time step."""

        # Four possible situations: cell completely relaxed & not defective;
        # cell completely relaxed and defective; cell ablated (permanently
        # unexcitable); cell still refractory (temporarily unexcitable).

        self.excitable_cells = np.ones(self.shape)

        # Defective
        self.excitable_cells[self._defective_cells] = np.random.random(np.sum(self._defective_cells))
        self.excitable_cells = self.excitable_cells > self._epsilon
        # Refractory
        self.excitable_cells[self.counts_until_relaxed != 0] = False
        # Ablated
        self.excitable_cells[self.permanently_unexcitable == 1] = False
        
        # Remaining cells are relaxed and have no reason to not excite given
        # a stimulus

    def __update_counts_until_relaxed(self):

        refractory_cells = self.counts_until_relaxed > 0
        self.counts_until_relaxed[refractory_cells] -= 1
        self.counts_until_relaxed[self.wavefront] = self._refractory_period 

    def determine_lateral_couplings(self, nu):

        self.couplings_set = True
        self._nu = nu
        self.lateral_couplings = self.__random_lateral <= float(nu)

    def evolve(self, pulse=False):

        if self.couplings_set is False:
            raise ValueError("""Value for fraction of lateral couplings present 
                                must be set.""")

        self.__find_cells_to_excite()

        new_wavefront = np.roll(self.wavefront * self.lateral_couplings, -1, axis=0)
        new_wavefront += np.roll(self.wavefront, 1, axis=0) * self.lateral_couplings
        new_wavefront[:, :-1] += self.wavefront[:, 1:]
        new_wavefront[:, 1:] += self.wavefront[:, :-1]
        new_wavefront *= self.excitable_cells

        if pulse is True:
            new_wavefront[:, 0] += self.excitable_cells[:, 0]

        self.wavefront = new_wavefront

        self.__update_counts_until_relaxed()

    def render_unexcitable(self, one_corner, another_corner):
        
        little_x = int(min(one_corner[0], another_corner[0]))
        big_x = int(max(one_corner[0], another_corner[0]))
        little_y = int(min(one_corner[1], another_corner[1]))
        big_y = int(max(one_corner[1], another_corner[1]))
        self.permanently_unexcitable[little_y:big_y, little_x:big_x] = 1

    def number_of_active_cells(self):

        return np.sum(self.wavefront)

    def reset(self):

        """Return Myocardium to resting state everywhere."""

        self.counts_until_relaxed[:, :] = 0
        self.wavefront[:, :] = False

