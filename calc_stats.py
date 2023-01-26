import glob
import natsort

import numpy as np
import nibabel as nib
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# import gtda
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from gtda.diagrams import Amplitude




# Sort the control and patient files within their respective folders
sorted = natsort.natsorted
files = [sorted(glob.glob('/home/User/MRI_Data/Control/*nii')),
         sorted(glob.glob('/home/User/MRI_Data/Patient/*nii'))]


def calc_stats(fig, files, name, threshold, max_edge_length):
    # Define a function that reads in the files, both control and patient scans
    # and where a threshold can be set. Returns a figure with the topological
    # statistics plotted against each other for the first three homology groups
    pe_stats = []
    amp_stats = []
    for num, file in enumerate(files):
        # num = int(file.split(name)[-1].split('.nii')[0])

        # Load in the file
        img = nib.load(file)
        victim_memmap = img.get_fdata()

        # Convert the data into a numpy array with the same shape
        victim = np.empty(victim_memmap.shape)
        victim[:] = victim_memmap[:]

        # print(np.quantile(victim[victim != 0.], [0.25, 0.5, 0.75]))
        threshold = threshold

        # Set a threshold that removes `dark` or `dark` data from the point
        # cloud
        arr_vic = np.argwhere(victim > threshold)

        # Construct a Vietoris-Rips simplicial complex on the data
        persistence = VietorisRipsPersistence(
            homology_dimensions = [0, 1, 2],
            collapse_edges = True,
            max_edge_length = max_edge_length,
            # plot every 10th point in the point cloud for quicker computation
        ).fit_transform(arr_vic[::10][None, :, :])

        # Calculate the persistent entropy and amplitude
        pe = PersistenceEntropy(normalize = True).fit_transform(persistence)
        amp = Amplitude().fit_transform(persistence)

        # Append them to a list
        pe_stats.append(pe)
        amp_stats.append(amp)


    # Setting up the plotting
    # convert lists of statistics into an array
    pe_stats = np.asarray(pe_stats)
    amp_stats = np.asarray(amp_stats)

    # Create 1D array of stats
    pe_arr = np.concatenate(pe_stats[:, 0].T)
    amp_arr = np.concatenate(amp_stats[:, 0].T)

    # Create matching arrays of homolgy group values
    hom0 = np.zeros(pe_stats[:, 0][:, 0].shape)
    hom1 = np.ones(pe_stats[:, 0][:, 1].shape)
    hom2 = np.full(pe_stats[:, 0][:, 2].shape, 2.)
    hom_arr = np.concatenate((hom0, hom1, hom2))

    # Plot the persistent entropy stats in 2D
    fig.add_trace(
        go.Scatter(
            mode = 'markers',
            x = hom_arr,
            y = pe_arr,
            name = name,
        ),
        row = 1,
        col = 1,
    )

    # Plot the amplitude stats in 2D
    fig.add_trace(
        go.Scatter(
            mode = 'markers',
            x = hom_arr,
            y = amp_arr,
            name = name,
        ),
        row = 1,
        col = 2,
    )

    # Plot the persistent entropy stats in 3D
    fig.add_trace(
        go.Scatter3d(
            mode = 'markers',
            x = pe_stats[:, 0][:, 0],
            y = pe_stats[:, 0][:, 1],
            z = pe_stats[:, 0][:, 2],
            name = name,
            marker = dict(size = 4),
        ),
        row = 2,
        col = 1,
    )
    fig.update_layout(scene = dict(
        xaxis_title = 'H0',
        yaxis_title = 'H1',
        zaxis_title = 'H2',
    ))

    # Plot the amplitude stats in 3D
    fig.add_trace(
        go.Scatter3d(
            mode = 'markers',
            x = amp_stats[:, 0][:, 0],
            y = amp_stats[:, 0][:, 1],
            z = amp_stats[:, 0][:, 2],
            name = name,
            marker = dict(size = 4),
        ),
        row = 2,
        col = 2,
    )


    # Save the pe_stats and amp_stats arrays to a csv file
    pd.DataFrame(pe_stats[:, 0]).to_csv(
        "/home/ADF/jas653/Prog/MRI/Plots/" + str(name) + "_" + str(threshold) +
        "_pe_stats.csv")
    pd.DataFrame(amp_stats[:, 0]).to_csv(
        "/home/ADF/jas653/Prog/MRI/Plots/" + str(name) + "_" + str(threshold) +
        "_amp_stats.csv")

    return fig, pe_stats, amp_stats


# Plotting: first create a subplot
fig = make_subplots(rows = 2, cols = 2,
                    specs = [[{'type': 'xy'}, {'type': 'xy'}],
                             [{'type': 'scene'}, {'type': 'scene'}]],
                    subplot_titles = ('2D Persistent Entropy on MRI Scans',
                                      '2D Amplitude on MRI Scans',
                                      '3D Persistent Entropy on MRI Scans',
                                      '3D Amplitude on MRI Scans,'
                                      ),
                    )

fig.update_layout(template = 'plotly_white',
                  font = dict(family = 'computer modern'))
# Update axis titles
fig.update_xaxes(title_text = 'Homolgy Group', row = 1, col = 1)
fig.update_yaxes(title_text = 'Value [-]', row = 1, col = 1)
fig.update_xaxes(title_text = 'Homolgy Group', row = 1, col = 2)
fig.update_yaxes(title_text = 'Value [-]', row = 1, col = 2)

fig.update_layout(scene = dict(
    xaxis_title = 'H0',
    yaxis_title = 'H1',
    zaxis_title = 'H2'),
    scene2 = dict(
        xaxis_title = 'H0',
        yaxis_title = 'H1',
        zaxis_title = 'H2'),
)


# Plot the values for both the control and patient scans for, say, a threshold
# of 100 and max_edge_distance of 5
fig, pe_con, amp_con = calc_stats(fig, files[0], 'Control', 100, 5)
fig, pe_pat, amp_pat = calc_stats(fig, files[1], 'Patient', 100, 5)

# Show the figure and save as a html in a local Plots folder
fig.show()
fig.write_html("/home/User/MRI/Plots/threshold_100.html")
