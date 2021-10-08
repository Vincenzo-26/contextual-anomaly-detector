# import from default libraries and packages
import datetime                         # data
import os                               # OS handling utils

import matplotlib.dates as mdates       # handle dates
import matplotlib.pyplot as plt         # plots
import numpy as np                      # general data manipulation
import pandas as pd                     # dataframe handling
from matplotlib import rc               # font plot

# import from the local module distancematrix
from distancematrix.calculator import AnytimeCalculator
from distancematrix.consumer import MatrixProfileLR, ContextualMatrixProfile
from distancematrix.consumer.contextmanager import GeneralStaticManager
from distancematrix.generator import Euclidean

# import from custom modules useful functions
from utils_functions import roundup, hour_to_dec, dec_to_hour, nan_diag, dec_to_obs
from anomaly_detection_functions import anomaly_detection

########################################################################################
# define a begin time to evaluate execution time & performance of algorithm
begin_time = datetime.datetime.now()
print('\n*********************\n' +
      'RUNNING Polito_Usecase.py\n' +
      'START: ' + begin_time.strftime("%Y-%m-%d %H:%M:%S"))

# useful paths
path_to_data = os.getcwd() + os.sep + 'Polito_Usecase' + os.sep + 'data' + os.sep
path_to_figures = os.getcwd() + os.sep + 'Polito_Usecase' + os.sep + 'figures' + os.sep

# figures variables
color_palette = 'viridis'
dpi_resolution = 300
fontsize = 10
line_style_context = "-"
line_style_other = ":"
line_color_context = "#D83C3B" # previously red
line_color_other = "#D5D5E0" # previously gray
# plt.style.use("seaborn-paper")
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Poppins']})
plt.rcParams.update({'font.size': fontsize})

########################################################################################
# load dataset
data = pd.read_csv(path_to_data + "polito.csv", index_col='timestamp', parse_dates=True)
obs_per_day = 96        # [observation/day]
obs_per_hour = 4        # [observation/hour]

min_power = 0           # [kW] minimum value of power
max_power = 850         # [kW] roundup(max(data.values)[0], 10)  # maximum value of power

ticks_power = list(range(min_power, max_power, roundup(max_power / 6, digit=100)))

position_x = 6          # [kW] position of day annotation on x axis
position_y = 750        # [kW] position of day annotation on y axis

# print dataset main characteristics
print('\n*********************\n',
      'DATASET: Electrical Load dataset from Substation C\n',
      '- From\t', data.index[0], '\n',
      '- To\t', data.index[len(data) - 1], '\n',
      '-', len(data.index[::obs_per_day]), '\tdays\n',
      '- 1 \tobservations every 15 min\n',
      '-', obs_per_day, '\tobservations per day\n',
      '-', obs_per_hour, '\tobservations per hour\n',
      '-', len(data), 'observations'
      )

'''
# Visualise the data
plt.figure(figsize=(10, 4))

plt.subplot(2, 1, 1)
plt.title("Total Electrical Load (complete)")
plt.plot(data)
plt.ylabel("Power [kW]")
plt.gca().set_ylim([min_power, max_power])
plt.gca().set_yticks(ticks_power)

plt.subplot(2, 1, 2)
plt.title("Total Electrical Load (first two weeks)")
plt.plot(data.iloc[:4 * 24 * 7 * 2])
plt.ylabel("Power [kW]")
plt.gca().set_ylim([min_power, max_power])
plt.gca().set_yticks(ticks_power)

plt.gca().xaxis.set_major_locator(mdates.DayLocator([1, 8, 15]))
plt.gca().xaxis.set_minor_locator(mdates.DayLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

plt.grid(b=True, axis="x", which='both', color='black', linestyle=':')

# add day labels on plot
for i in range(14):
    timestamp = data.index[position_x + i * obs_per_day]
    plt.text(timestamp, position_y, timestamp.day_name()[:3])

plt.tight_layout()

# save figure to plot directories
plt.savefig(path_to_figures + "dataset_lineplot.png", dpi=dpi_resolution, bbox_inches='tight')
plt.close()
'''

########################################################################################
# Define configuration for the Contextual Matrix Profile calculation.

# The number of time window has been selected from CART on total electrical power,
# results are contained in 'time_window.csv' file
time_window = pd.read_csv(path_to_data + "time_window.csv")

# The context is defined as 1 hour before time window, to be consistend with other analysis,
# results are loaded from 'm_context.csv' file
m_context = pd.read_csv(path_to_data + "m_context.csv")["m_context"][0]

# Define output file as dataframe
# in this file the anomaly results will be saved
df_output_all = pd.DataFrame()

# begin for loop on the number of time windows
for u in range(len(time_window)):

    ########################################################################################
    # Data Driven Context Definition
    if u == 0:
        # manually define context if it is the beginning
        context_start = 0                                   # [hours] i.e., 00:00
        context_end = context_start + m_context             # [hours] i.e., 01:00
        # [observations] = ([hour]-[hour])*[observations/hour]
        m = int((hour_to_dec(time_window["to"][u]) - m_context) * obs_per_hour)
    else:
        m = time_window["observations"][u]                  # [observations]
        context_end = hour_to_dec(time_window["from"][u])   # [hours]
        context_start = context_end - m_context             # [hours]

    '''
    # 2) User Defined Context
    # We want to find all the subsequences that start from 00:00 to 02:00 (2 hours) and covers the whole day
    # In order to avoid overlapping we define the window length as the whole day of
    observation minus the context length.

    # - Beginning of the context 00:00 AM [hours]
    context_start = 17

    # - End of the context 02:00 AM [hours]
    context_end = 19

    # - Context time window length 2 [hours]
    m_context = context_end - context_start  # 2

    # - Time window length [observations]
    # m = 96 [observations] - 4 [observation/hour] * 2 [hours] = 88 [observations] = 22 [hours]
    # m = obs_per_day - obs_per_hour * m_context
    m = 20 # with guess
    '''

    # print string to explain the created context in an intelligible way
    context_string = 'Subsequences of ' + dec_to_hour(m / obs_per_hour) + ' h that starts between ' + \
                     dec_to_hour(context_start) + ' and ' + dec_to_hour(context_end)

    # contracted context string for names
    context_string_small = 'ctx_from' + dec_to_hour(context_start) + \
                           '_to' + dec_to_hour(context_end) + "_m" + dec_to_hour(m / obs_per_hour)
    # remove : to resolve path issues
    context_string_small = context_string_small.replace(":", "_")

    print('\n*********************\n', 'CONTEXT: ' + context_string + " (" + context_string_small + ")")

    # if figures directory doesnt exists create and save into it
    if not os.path.exists(path_to_figures + context_string_small):
        os.makedirs(path_to_figures + context_string_small)

    '''
    # Context Definition:
    # example FROM 00:00 to 02:00
    # - m_context = 2 [hours]
    # - obs_per_hour = 4 [observations/hour]
    # - context_start = 0 [hours]
    # - context_end = context_start + m_context = 0 [hours] + 2 [hours] = 2 [hours]
    contexts = GeneralStaticManager([
        range(
            # FROM  [observations]  = x * 96 [observations] + 0 [hour] * 4 [observation/hour]
            (x * obs_per_day) + context_start * obs_per_hour,
            # TO    [observations]  = x * 96 [observations] + (0 [hour] + 2 [hour]) * 4 [observation/hour]
            (x * obs_per_day) + (context_start + m_context) * obs_per_hour)
        for x in range(len(data) // obs_per_day)
    ])
    '''

    # Context Definition:
    contexts = GeneralStaticManager([
        range(
            # FROM  [observations]  = x * 96 [observations] + 0 [hour] * 4 [observation/hour]
            (x * obs_per_day) + dec_to_obs(context_start, obs_per_hour),
            # TO    [observations]  = x * 96 [observations] + (0 [hour] + 2 [hour]) * 4 [observation/hour]
            (x * obs_per_day) + dec_to_obs(context_end, obs_per_hour))
        for x in range(len(data) // obs_per_day)
    ])

    ########################################################################################
    # Calculate Contextual Matrix Profile
    calc = AnytimeCalculator(m, data.values.T)

    # Add generator Not Normalized Euclidean Distance
    distance_string = 'Not Normalized Euclidean Distance'
    calc.add_generator(0, Euclidean())

    # We want to calculate CMP initialize element
    cmp = calc.add_consumer([0], ContextualMatrixProfile(contexts))

    # Calculate Contextual Matrix Profile (CMP)
    calc.calculate_columns(print_progress=True)
    print("\n")

    # if data directory doesnt exists create and save into it
    if not os.path.exists(path_to_data + context_string_small):
        os.makedirs(path_to_data + context_string_small)

    # Save CMP for R plot
    np.savetxt(path_to_data + context_string_small + os.sep + 'plot_cmp_full.csv',
               nan_diag(cmp.distance_matrix),
               delimiter=",")

    '''
    # calculate the date labels to define the extent of figure
    date_labels = mdates.date2num(data.index[::m].values)
    # plot CMP as matrix
    plt.figure(figsize=(10, 10))

    extents = [date_labels[0], date_labels[-1], date_labels[0], date_labels[-1]]
    CMP_plot(contextual_mp=cmp.distance_matrix,
             palette=color_palette,
             title='Contextual Matrix Profile',
             extent=extents,
             legend_label=distance_string,
             date_ticks=14 * 2
             )

    plt.savefig(path_to_figures + context_string_small + os.sep + "cmp_context.png",
                dpi=dpi_resolution,
                bbox_inches='tight')
    plt.close()
    '''

    ########################################################################################
    # Load Cluster results as boolean dataframe: each column represents a group
    annotation_df = pd.read_csv(path_to_data + "group_cluster.csv", index_col='timestamp', parse_dates=True)
    # initialize dataframe of results for context to be appended to the overall result
    df_output_context = annotation_df.astype(int)
    # set labels
    day_labels = data.index[::obs_per_day]
    # get number of groups
    n_group = annotation_df.shape[1]

    # perform analysis of context on groups
    for i in range(n_group):

        # time when computation starts
        begin_time_group = datetime.datetime.now()

        # get group name from dataframe
        group_name = annotation_df.columns[i]

        # add column of context of group in df_output
        df_output_context[group_name + "." + context_string_small] = [0 for i in range(len(df_output_context))]

        # if figures directory doesnt exists create and save into it
        if not os.path.exists(path_to_figures + context_string_small + os.sep + group_name):
            os.makedirs(path_to_figures + context_string_small + os.sep + group_name)

        # create empty group vector
        group = np.array(annotation_df.T)[i]
        # get cmp from previously computed cmp
        group_cmp = cmp.distance_matrix[:, group][group, :]
        # substitute inf with zeros
        group_cmp[group_cmp == np.inf] = 0
        # get dates
        group_dates = data.index[::obs_per_day].values[group]

        # save group CMP for R plot
        np.savetxt(path_to_data + context_string_small + os.sep + 'plot_cmp_' + group_name + '.csv',
                   nan_diag(group_cmp), delimiter=",")

        '''
        # plot CMP as matrix
        plt.figure(figsize=(7, 7))
        CMP_plot(contextual_mp=group_cmp,
                 palette=color_palette,
                 title="Power CMP (" + group_name + " only)",
                 xlabel=group_name + " Index",
                 legend_label=distance_string
                 )
        plt.savefig(path_to_figures + context_string_small + os.sep + group_name + os.sep + "polito_cmp.png",
                    dpi=dpi_resolution,
                    bbox_inches='tight')
        plt.close()
        '''

        #######################################
        # calculate anomaly score trhough majority voting
        cmp_ad_score = anomaly_detection(group=group, group_cmp=group_cmp)
        # set to nan if zero (no anomaly)
        cmp_ad_score = np.where(cmp_ad_score == 0, np.nan, cmp_ad_score)

        # the number of anomalies is the number of non nan elements, count
        num_anomalies_to_show = np.count_nonzero(~np.isnan(cmp_ad_score))

        # Ordering of all days, from most to least anomalous in order of severity
        ad_order = np.argsort(cmp_ad_score)[::-1]

        # move na at the end of the vector
        ad_order = np.roll(ad_order, -np.count_nonzero(np.isnan(cmp_ad_score)))

        # create a vector to plot correctly the graph
        cmp_ad_score_plot = cmp_ad_score[ad_order][0:num_anomalies_to_show]

        # only visualize if some anomaly are shown
        if num_anomalies_to_show > 0:

            # Visualise the top anomalies according to the CMP
            fig, ax = plt.subplots(num_anomalies_to_show, 2,
                                   sharex='all',
                                   #sharey='all',
                                   figsize=(10, 14 / 8 * num_anomalies_to_show),
                                   #gridspec_kw={'wspace': 0., 'hspace': 0.}
                                   )
            fig.suptitle("Anomaly Detection " + group_name.replace("_", " "))


            for j in range(num_anomalies_to_show):
                anomaly_index = ad_order[j]
                anomaly_range = range(obs_per_day * anomaly_index, obs_per_day * (anomaly_index + 1))
                date = day_labels[anomaly_index]

                # update output dataframe and add severity
                df_output_context.loc[df_output_context.index.values == np.datetime64(date),
                                      group_name + "." + context_string_small] = cmp_ad_score[anomaly_index]

                # dataframe for group power and energy
                power_group = data.values.reshape((-1, obs_per_day))[group].T
                energy_group = np.empty((power_group.shape[0], power_group.shape[1]))
                for k in range(0, power_group.shape[1]):
                    energy_group[:, k] = np.cumsum(power_group[:, k])

                # dataframe for group power and energy for anomaly
                power_group_anomaly = data.values[anomaly_range]
                energy_group_anomaly = np.empty((power_group_anomaly.shape[0], power_group_anomaly.shape[1]))
                for k in range(0, power_group_anomaly.shape[1]):
                    energy_group_anomaly[:, k] = np.cumsum(power_group_anomaly[:, k])

                ax[j, 0].plot(energy_group,
                              c=line_color_other,
                              alpha=0.3)
                ax[j, 0].plot(
                    range(dec_to_obs(context_start, obs_per_hour), (dec_to_obs(context_end, obs_per_hour) + m)),
                    energy_group_anomaly[dec_to_obs(context_start, obs_per_hour):(dec_to_obs(context_end, obs_per_hour) + m)],
                    c=line_color_context,
                    linestyle=line_style_context)
                ax[j, 0].plot(energy_group_anomaly,
                              c=line_color_context,
                              linestyle=line_style_other)
                ax[j, 0].set_title("Anomaly " + str(j + 1) + " - Severity " +  str(int(cmp_ad_score[anomaly_index])) )


                ax[j, 1].plot(power_group,
                              c=line_color_other,
                              alpha=0.3)
                ax[j, 1].plot(
                    range(dec_to_obs(context_start, obs_per_hour), (dec_to_obs(context_end, obs_per_hour) + m)),
                    power_group_anomaly[
                    dec_to_obs(context_start, obs_per_hour):(dec_to_obs(context_end, obs_per_hour) + m)],
                    c=line_color_context,
                    linestyle=line_style_context)
                ax[j, 1].plot(power_group_anomaly,
                              c=line_color_context,
                              linestyle=line_style_other)
                ax[j, 1].set_ylim([min_power, max_power])
                ax[j, 1].set_yticks(ticks_power)
                ax[j, 1].set_title(date.day_name() + " " + str(date)[:10])


            ax[0, 0].set_xticks(range(0, 97, 24))
            ticklabels = ["{hour}:00".format(hour=(x // obs_per_hour)) for x in range(0, 97, 24)]
            # ticklabels[-1] = ""
            ax[0, 0].set_xticklabels(ticklabels)

            plt.tight_layout()

            # ax[num_anomalies_to_show // 2, 0].set_ylabel("Power [kW]")
            # ax[num_anomalies_to_show - 1, 1].set_xlabel("Time of day")

            plt.savefig(path_to_figures + context_string_small + os.sep + group_name + os.sep + "polito_anomalies.png",
                        dpi=dpi_resolution,
                        bbox_inches='tight')
            plt.close()

            # print the execution time
            time_interval_group = datetime.datetime.now() - begin_time_group
            hours, remainder = divmod(time_interval_group.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            print("- " + group_name + ' (' + str(int(seconds)) + ' s' + ') -> ' +
                  str(num_anomalies_to_show) + ' anomalies')

        # if no anomaly to show not visualize
        else:
            pass

    # at the end of loop on groups save dataframe corresponding to given context or append to existing one
    if df_output_all.empty:
        df_output_all = df_output_context
    else:
        df_output_all = pd.concat([df_output_all, df_output_context], axis=1)

# at the end of loop on context save dataframe of results
df_output_all.to_csv(path_to_data + "anomaly_results.csv")

# print the execution time
total_time = datetime.datetime.now() - begin_time
hours, remainder = divmod(total_time.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)
print('\n*********************\n' + "END: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("TOTAL " + str(int(minutes)) + ' min ' + str(int(seconds)) + ' s')
