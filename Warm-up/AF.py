#### Copyright Joel Dyer, 29/07/2016 ####


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from AFModel import Myocardium

class TimeTracker:

	def __init__(self, tmax=np.inf):

		self.tmax = tmax
		self.tinit = 0

	def __iter__(self):
		
		t = 1
		while t < self.tmax:
			yield t
			if self.stop is True:
				raise StopIteration()
			t += 1

	def stop(self):

		self.stop = True

def run(tmax=np.inf, heart_rate=250, tissue_shape=(200, 200), nu=0.8, delta=0.05, p=0.95, refractory_period=50):

	ax = plt.gca()
	s = Myocardium(tissue_shape, nu, delta, p, refractory_period)
	qm = ax.pcolorfast(s.counts_until_relaxed)
	qm.set_array(s.counts_until_relaxed)
	ax.set_title('0')
	tt = TimeTracker()
	plt.draw()
	plt.pause(0.01)

	fig = ax.get_figure()

	def handle_close(evt):
		tt.stop()

	fig.canvas.mpl_connect('close_event', handle_close)

	for time in tt:
		s.evolve_wavefront()
		s.update_counts_until_relaxed()
		if time%heart_rate == 0:
			s.pulse()
			qm.set_array(s.counts_until_relaxed)
			ax.set_title('{}'.format(time))
			plt.draw()
		else:
			qm.set_array(s.counts_until_relaxed)
			ax.set_title('{}'.format(time))
			plt.draw()
		plt.pause(0.01)
