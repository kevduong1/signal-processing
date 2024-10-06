'''
Practice example of signal processing

reference: https://towardsdatascience.com/hands-on-signal-processing-with-python-9bda8aad39de
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import pywt
import os
current_directory = os.getcwd()

#======================================================================
# Define helper functions
#======================================================================
def clear_plot_params():
    # Clear previous plotting parameters
    plt.close()
    plt.clf()
    plt.cla()

#======================================================================
# pre-process and plot raw data
#======================================================================
# If script is ran from root directory of the repo, join paths to the data folder
csv_file_path = os.path.join( current_directory, 'data\\AEP_hourly.csv' )

# set x and y data
data_fft = pd.read_csv( csv_file_path, encoding="utf-8" )
y = np.array( data_fft.AEP_MW )
x = data_fft.index
date_array = pd.to_datetime( data_fft.Datetime )

# Plot raw data
plt.plot( date_array, y )
plt.xlabel( 'Date', fontsize=20 )
plt.ylabel( 'MW Energy Consumption', fontsize=20 )
# plt.show()

# Remove underlying trend to make it easier to analyze patterns and fluctuations
y_detrend = signal.detrend( y )

clear_plot_params()
plt.plot( date_array, y_detrend, color='firebrick', label='Detrended Signal' )
plt.plot( date_array, y, color='navy', label='Raw Signal')
plt.legend()
plt.xlabel('Date',fontsize=20)
plt.ylabel('Temperature',fontsize=20)
# plt.show

#======================================================================
# Perform frequency Analysis
#======================================================================
# Call fast fourier transform function
FFT = np.fft.fft( y_detrend )

# The FFT of a real-valued signal is symmetric, with negative frequencies 
# mirroring the positive ones. We only need the first half (positive frequencies)
# for analysis, as the negative half doesn't add new information.
new_N = int( len( FFT ) / 2 )

# Define natural frequency
f_nat = 1

# Create an array of evenly spaced frequency values between a very small number (close to 0) 
# and half the natural frequency (f_nat / 2). This represents the positive frequency range.
new_X = np.linspace( 10 ** -12, f_nat / 2, new_N, endpoint=True )

# Convert frequencies to periods
new_Xph = 1.0 / ( new_X )

# Calculate the magnitude of the FFT components
FFT_abs = np.abs( FFT )

clear_plot_params()

plt.plot( new_Xph, 2 * FFT_abs[ 0 : int( len( FFT ) / 2. ) ] / len( new_Xph ), color='black' )
plt.xlabel( 'Period ($h$)', fontsize=20 )
plt.ylabel( 'Amplitude', fontsize=20 )
plt.title( '(Fast) Fourier Transform Method Algorithm', fontsize=20 )
plt.grid( True )
plt.xlim( 0, 200 )

# Plot frequency domain
plt.show()
'''
anaylsis

There's a pattern of reacurring usage every 12 hour, 24 hour, 3 day, and 7 day periods

analysis of frequencies:
7 days: some reasoning for patterns of high energy consumptions repeating every 7 days could be due
        to people doing weekly errands such as doing laundry or running the dishwasher a certain day of every week
        which consumes a lot of energy and can lead to spikes a certain day of the week
        
3 days: similar to the weekly occurence, except on a shorter scale for maybe bigger families that have to run
        those machines more frequently. Also could be something to do with industrial machines being used at
        businesses every few days
        
24 hours: Could represent frequent energy use at a certain time of day like running the coffee maker at a specific
          time every morning, or turning on lights at a certain time.

12 hours: Could represent shifts in energy use for example day vs night time.
'''


fft_abs = 2 * FFT_abs[ 0 : int( len( FFT ) / 2. ) ] / len( new_Xph )
fft_abs = pd.DataFrame( fft_abs, columns = [ 'Amplitude' ] )
fft_sorted = fft_abs.sort_values( by='Amplitude', ascending=False ).head( 20 )

print( fft_sorted )


#======================================================================
# Noise filtering 
#======================================================================
# Defining the filtering function
def fft_filter( th ):
    fft_tof = FFT.copy()
    fft_tof_abs = np.abs( fft_tof )
    fft_tof_abs = 2 * fft_tof_abs / len( new_Xph )
    fft_tof[ fft_tof_abs <= th ] = 0
    return fft_tof

# Showing the plots at different thresholds values
# Defining the amplitude filtering function
def fft_filter_amp( th ):
    fft_tof = FFT.copy()
    fft_tof_abs = np.abs( fft_tof )
    fft_tof_abs= 2 * fft_tof_abs / len( new_Xph )
    fft_tof_abs[ fft_tof_abs <= th ] = 0
    return fft_tof_abs[ 0 : int( len( fft_tof_abs ) / 2. ) ]

K_plot = [ 10, 200, 700, 1500 ]
j = 0

clear_plot_params()

for k in K_plot:
    j = j + 1
    plt.subplot( 2, 2, j )
    plt.title( 'k=%i' % ( k ) )
    plt.xlim( 0, 200 )
    plt.plot( new_Xph, 2 * FFT_abs[ 0 : int( len( FFT ) / 2. ) ] /len( new_Xph ), color='navy', alpha=0.5, label='Original' )
    plt.grid( True )
    plt.plot( new_Xph, fft_filter_amp( k ), 'red', label='Filtered' )
    plt.xlabel( 'Time($h$)' )
    plt.ylabel( 'Amplitude' )
    plt.legend()
plt.subplots_adjust( hspace = 0.5 )
plt.show()