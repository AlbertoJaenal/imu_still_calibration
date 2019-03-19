'''
/*****************************************************************************
**         IMU still calibration for intrinsic parameter estimation         **
******************************************************************************
**                                                                          **
**  Copyright(c) 2019, Alberto Jaenal Galvez, University of Malaga          **
**  Copyright(c) 2019, MAPIR group, University of Malaga                    **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/
'''

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('numpy_path', help='Path to a .npy file for the dataset')

    parser.add_argument("-f", "--frequency", help="Sampling frequency (default: 200.0 Hz)", type=float, default=200.0)
    parser.add_argument("-w", "--windows", help="Minimum windows for the Tau loop (default: 30)", type=int, default=30)
    parser.add_argument("-n", "--num_samples", help="Number of samples for the Allan Computation(default: 2200)",
                        type=int, default=2200)
    parser.add_argument("-o", "--devOrder", help="Device saving order: 0 - Gyro/Acc (default), 1 - Acc/Gyro", type=int,
                        default=0)

    parser.add_argument("-a", "--accVars", help="Accelerometer variables used for the printing. Ex: xyz, xz",
                        default='xyz')
    parser.add_argument("-g", "--gyroVars", help="Gyroscope variables used for the printing. Ex: xyz, xz",
                        default='xyz')

    parser.add_argument("-v", "--verbose", action='store_true', default='xyz')

    parser.add_argument("--limWlow", help="Minimum integration time for the bias_w (default: 0.02 s)", type=float,
                        default=0.02)
    parser.add_argument("--limWupp", help="Maximum integration time for the bias_w (default: 1 s)", type=float,
                        default=1)
    parser.add_argument("--limBlow", help="Minimum integration time for the bias_b (default: 1000 s)", type=float,
                        default=1000.0)
    parser.add_argument("--limBupp", help="Maximum integration time for the bias_b (default: 6000 s)", type=float,
                        default=6000.0)
    args = parser.parse_args()


    ##################################################
    ### ALLAN VARIANCE COMPUTATION                 ###
    ## Check if the file has been previously crated ##
    ##################################################

    TAU_0 = 1 / args.frequency
    num = args.num_samples
    # For each [x,y,z]
    # Example: timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z, (temperature)
    if args.devOrder == 0:
        print(' \nSetting Gyroscope-Accelerometer as data order...\n')
        DATA_AXES = {'Acc': [4, 5, 6], 'Gyro': [1, 2, 3]}
    else:
        print(' \nSetting Accelerometer-Gyroscope as data order...\n')
        DATA_AXES = {'Acc': [1, 2, 3], 'Gyro': [4, 5, 6]}

    if not os.path.exists(args.numpy_path.replace('.npy', '_AllanVar.npy')):
        data = np.load(args.numpy_path)
        M = data.shape[0]

        ## Loop to obtain a logarithmic scale for the x axis with num non-repeated numbers
        f = True
        n = num
        while f:
            f = (len(np.unique(np.logspace(0, np.log10((M - args.windows) // 2), num=n, dtype=np.int))) < num)
            n += 10

        entries = np.unique(np.logspace(0, np.log10((M - args.windows) // 2), num=n, dtype=np.int))
        num = len(entries)

        ###############################
        # Tau (integration time) loop #
        ###############################
        variances = np.empty([num, 7])
        # The variables must be placed from the axis 1. The axis 0 is reserved for the timestamps (not used in this algorithm)
        ret = np.cumsum(data[:, 1:7], axis=0)
        # Deletes the data in order to save space
        del data
        f = time.time()

        for i, n in enumerate(entries):
            # Optimum OVERLAPPING computation
            variances[i, 1:] = 1 / (2 * (M - 2 * n + 1)) * (
                    (ret[2 * n:, :] - 2 * ret[n:-n, :] + ret[:-2 * n, :]) ** 2).sum(axis=0) / (n ** 2)
            if args.verbose:
                print(i, n, (M - args.windows) // 2, time.time() - f, )
            f = time.time()

        # Store also the integration numbers (n), not tau
        variances[:, 0] = entries[:]
        np.save(args.numpy_path.replace('.npy', '_AllanVar.npy'), variances)

        # Deletes the data in order to save space
        del ret
    else:
        variances = np.load(args.numpy_path.replace('.npy', '_AllanVar.npy'))

    #########################################
    ### BIASES COMPUTATION                ###
    #########################################
    # log(y) = m*log(x) + b                 #
    # Minimize: m*log(x) - log(y) = -b      #
    # SIGMA_W = 10^(-1/2*log(1)+optBias_W)  #
    # SIGMA_B = 10^(1/2*log(3)+optBias_B)   #
    #########################################
    # Obtain the indexes of the indicated time intervals
    indW = np.argwhere(np.logical_and(variances[:, 0] * TAU_0 <= args.limWupp, variances[:, 0] * TAU_0 >= args.limWlow))
    indB = np.argwhere(np.logical_and(variances[:, 0] * TAU_0 <= args.limBupp, variances[:, 0] * TAU_0 >= args.limBlow))

    # Create vectors with the indicated variables (x, y and/or z) for each device
    xAccW, yAccW, xAccB, yAccB = [], [], [], []
    for i, v in enumerate(['x', 'y', 'z']):
        if v in args.accVars:
            xAccW.append(variances[indW, 0])
            yAccW.append(variances[indW, DATA_AXES['Acc'][i]])
            xAccB.append(variances[indB, 0])
            yAccB.append(variances[indB, DATA_AXES['Acc'][i]])

    xGyroW, yGyroW, xGyroB, yGyroB = [], [], [], []
    for i, v in enumerate(['x', 'y', 'z']):
        if v in args.gyroVars:
            xGyroW.append(variances[indW, 0])
            yGyroW.append(variances[indW, DATA_AXES['Gyro'][i]])
            xGyroB.append(variances[indB, 0])
            yGyroB.append(variances[indB, DATA_AXES['Gyro'][i]])

    # Create the numpy arrays for the Least Squares problem
    # NOTE: it is necesary to obtain the Allan Deviation and the integration time in seconds
    xAccW = np.concatenate(xAccW) * TAU_0
    yAccW = np.sqrt(np.concatenate(yAccW))
    xAccB = np.concatenate(xAccB) * TAU_0
    yAccB = np.sqrt(np.concatenate(yAccB))

    xGyroW = np.concatenate(xGyroW) * TAU_0
    yGyroW = np.sqrt(np.concatenate(yGyroW))
    xGyroB = np.concatenate(xGyroB) * TAU_0
    yGyroB = np.sqrt(np.concatenate(yGyroB))

    # Obtain the biases by a minimization problem
    biasAccW = 10 ** (
    np.linalg.lstsq(-np.ones_like(xAccW), np.squeeze(np.log10(xAccW) * (-1 / 2) - np.log10(yAccW)))[0])
    biasAccB = 10 ** (np.linalg.lstsq(-np.ones_like(xAccB), np.squeeze(np.log10(xAccB) * (1 / 2) - np.log10(yAccB)))[
                          0] + np.log10(3) / 2)
    biasGyroW = 10 ** (
    np.linalg.lstsq(-np.ones_like(xGyroW), np.squeeze(np.log10(xGyroW) * (-1 / 2) - np.log10(yGyroW)))[0])
    biasGyroB = 10 ** (
            np.linalg.lstsq(-np.ones_like(xGyroB), np.squeeze(np.log10(xGyroB) * (1 / 2) - np.log10(yGyroB)))[
                0] + np.log10(3) / 2)

    # Print the results
    print('\n\n\nAcelerometer:\nSigma_w: ', biasAccW, '\nSigma_b: ', biasAccB)
    print('Gyroscope:\nSigma_w: ', biasGyroW, '\nSigma_b: ', biasGyroB, '\n\n\n')

    with open(args.numpy_path.replace('.npy','.yaml'), 'w') as f:
        f.write('#Accelerometers\n')
        f.write('accelerometer_noise_density: ' + str(biasAccW.squeeze()) + ' #Noise density (continuous-time)\n')
        f.write('accelerometer_random_walk: ' + str(biasAccB.squeeze()) + ' #Bias random walk\n\n')

        f.write('#Gyroscopes\n')
        f.write('gyroscope_noise_density: ' + str(biasGyroW.squeeze()) + ' #Noise density (continuous-time)\n')
        f.write('gyroscope_random_walk: ' + str(biasGyroB.squeeze()) + ' #Bias random walk\n\n')

        f.write('rostopic: /imu0/data #the IMU ROS topic\n')
        f.write('update_rate: ' + str(args.frequency) + ' #Hz (for discretization of the values above)\n')

    ############
    # PLOTTING #
    ############
    # X coordinates for the regressed lines
    xW = [args.limWlow, args.limWupp, 1]
    xB = [3, args.limBlow, args.limBupp]

    fig, ax = plt.subplots(nrows=2, sharex=True)

    # Plot the curves for the Accelerometer
    ax[0].plot(variances[:, 0] * TAU_0, np.sqrt(variances[:, DATA_AXES['Acc'][0]]), label='x')
    ax[0].plot(variances[:, 0] * TAU_0, np.sqrt(variances[:, DATA_AXES['Acc'][1]]), label='y')
    ax[0].plot(variances[:, 0] * TAU_0, np.sqrt(variances[:, DATA_AXES['Acc'][2]]), label='z')
    # Plot the biases
    ax[0].plot(1, biasAccW, 'g*', label='sig_b', markersize=10)  # ='+str(round(float(biasAccW),6)))
    ax[0].plot(3, biasAccB, 'r*', label='sig_w', markersize=10)  # ='+str(round(float(biasAccB),6)))
    # Plot the regressed lines
    ax[0].plot(np.array(xW), 10 ** (np.log10(biasAccW) - np.log10(xW) / 2), 'k', marker='o', linestyle='--',
               markersize=3)
    ax[0].plot(np.array(xB), 10 ** (np.log10(biasAccB) + np.log10(xB) / 2 - np.log10(3) / 2), 'k', marker='o',
               linestyle='--', markersize=3)
    # Plot settings
    ax[0].set_title('Accelerometer Allan Deviation')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].legend()

    # Plot the curves for the Gyroscope
    ax[1].plot(variances[:, 0] * TAU_0, np.sqrt(variances[:, DATA_AXES['Gyro'][0]]), label='x')
    ax[1].plot(variances[:, 0] * TAU_0, np.sqrt(variances[:, DATA_AXES['Gyro'][1]]), label='y')
    ax[1].plot(variances[:, 0] * TAU_0, np.sqrt(variances[:, DATA_AXES['Gyro'][2]]), label='z')
    # Plot the biases
    ax[1].plot(1, biasGyroW, 'g*', label='sig_b', markersize=10)  # ='+str(round(float(biasGyroW),6)))
    ax[1].plot(3, biasGyroB, 'r*', label='sig_w', markersize=10)  # ='+str(round(float(biasGyroB),6)))
    # Plot the regressed lines
    ax[1].plot(np.array(xW), 10 ** (np.log10(biasGyroW) - np.log10(xW) / 2), 'k', marker='o', linestyle='--',
               markersize=3)
    ax[1].plot(np.array(xB), 10 ** (np.log10(biasGyroB) + np.log10(xB) / 2 - np.log10(3) / 2), 'k', marker='o',
               linestyle='--', markersize=3)
    # Plot settings
    ax[1].set_title('Gyroscope Allan Deviation')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].grid()
    ax[1].legend()

    fig.savefig(args.numpy_path.replace('.npy', '.png'))
    plt.show()
