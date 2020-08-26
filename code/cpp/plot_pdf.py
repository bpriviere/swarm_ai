import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import math

import os, subprocess
from matplotlib.backends.backend_pdf import PdfPages 

## Inputs
data_path = "output.csv"
pdf_path  = "output.pdf"

## Load the data
data = np.loadtxt(data_path, delimiter=',')

## Create an animation
for ii in range(1, data.shape[0]):
	# Extract the data
	A1_x = data[0:ii,0]
	A1_y = data[0:ii,1]
	D1_x = data[0:ii,4]
	D1_y = data[0:ii,5]

	A1_v = math.sqrt(data[ii,2]**2+data[ii,3]**2)
	D1_v = math.sqrt(data[ii,6]**2+data[ii,7]**2)

	# Create plot
	fig, ax = plt.subplots()
	ax.axis('equal')

	ax.plot(A1_x, A1_y, label="attacker")
	ax.plot(D1_x, D1_y, label="defender")

	# Add text
	textBox = dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0.5)
	ax.text(0.1, 0.07, "{:.3f}".format(A1_v), transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center',  bbox=textBox,zorder=11)
	ax.text(0.9, 0.07, "{:.3f}".format(D1_v), transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=textBox,zorder=11)
	ax.text(0.1, 0.07, 'Velocity (Attacker)', transform=ax.transAxes, fontsize=6, verticalalignment='bottom', horizontalalignment='center',  bbox=textBox,zorder=11)
	ax.text(0.9, 0.07, 'Velocity (Defender)', transform=ax.transAxes, fontsize=6, verticalalignment='bottom', horizontalalignment='center',  bbox=textBox,zorder=11)

	# Format plot
	ax.set_xlim([0,0.5])
	ax.set_ylim([0,0.5])

	ax.legend()

## Save (and open) PDF
# Save PDF
fn = os.path.join( os.getcwd(), pdf_path)
pp = PdfPages(fn)
for i in plt.get_fignums():
	pp.savefig(plt.figure(i))
	plt.close(plt.figure(i))
pp.close()

# Open PDF
subprocess.call(["xdg-open", pdf_path])

