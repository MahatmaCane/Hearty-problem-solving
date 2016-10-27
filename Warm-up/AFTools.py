import numpy as np

class ECG:

    def __init__(self, x, y, shape):

        xs, ys = range(shape[0]), range(shape[1])
        xxs, yys = np.meshgrid(xs, ys)
        self.x_diffs = xxs - x
        self.y_diffs = yys - y

        # Vectorised function for getting TMPs
        self.__get_tmps = np.vectorize(self.__get_voltages) 

        # Vectorised function for getting denominator in ECG expression
        self.__get_denominators = np.vectorize(self.__get_denominator)
        self._denoms = self.__get_denominators(self.x_diffs, self.y_diffs)
        self._denoms[self._denoms == 0] = 1
        self.data = []

    def __get_voltages(self, myo_count, ref_per):

        """Not to be used directly but in vectorised form."""

        return 20 - 110*(1 - myo_count/float(ref_per))

    def __get_denominator(self, x_diff, y_diff):

        """Not to be used directly but in vectorised form."""

        return ((x_diff)**2 + (y_diff)**2 + 3**2)**(1.5)

    def get_ECG(self, myo_counts, ref_per):

        tmps = self.__get_tmps(myo_counts, ref_per)
        vx_diff = tmps.copy()
        vx_diff[:, 1:] -= vx_diff[:, :-1]

        vy_diff = tmps - np.roll(tmps, -1, axis=0)

        compound = (self.x_diffs * vx_diff + self.y_diffs * vy_diff)/self._denoms
        to_record = np.sum(compound)
        self.data.append(to_record)
