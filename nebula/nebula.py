"""
Embodied AI Engine Prototype AKA "Nebula". This object takes a live signal
(such as body tracking, or real-time sound analysis) and generates a response
that aims to be felt as co-creative. The response is a flow of neural network
emissions data packaged as a dictionary, and is gestural over time. This, when
plugged into a responding script (such as a sound generator, or QT graphics)
gives the feeling of the AI creating in-the-moment with the human in-the-loop.

© Craig Vear 2022-24
craig.vear@nottingham.ac.uk

Dedicated to Fabrizio Poltronieri
"""
import logging
import numpy as np
import warnings
from scipy import signal
from threading import Thread
from time import sleep, time
from random import random, choice

import config
from modules.bitalino import BITalino
# from modules.brainbit import BrainbitReader
from modules.listener import Listener
from nebula.ai_factory import AIFactoryRAMI


def scaler(in_feature, mins, maxs):
    """
    Min-max scaler with clipping.
    """
    warnings.filterwarnings('error')
    in_feature = np.array(in_feature)
    mins = np.array(mins)
    maxs = np.array(maxs)
    try:
        norm_feature = (in_feature - mins) / (maxs - mins)
    except RuntimeWarning:
        logging.info("Scaler encountered zero division")
        norm_feature = in_feature
    norm_feature = norm_feature.clip(0, 1)
    warnings.simplefilter("always")
    return norm_feature


class Nebula(Listener, AIFactoryRAMI):
    """
    Nebula is the core "director" of an AI factory. It generates data in
    response to incoming percepts from human-in-the-loop interactions, and
    responds in-the-moment to the gestural input of live data.
    There are 4 components:
        - Nebula: as "director" it coordinates the overall operations of the AI
        Factory.
        - AIFactory: builds the neural nets that form the factory, coordinates
        data exchange, and liases with the common data dict.
        - Hivemind: is the central dataclass that holds and shares all the data
        exchanges in the AI factory.
        - Conducter: receives the live percept input from the client and
        produces an affectual response to it's energy input, which in turn
        interferes with the data generation.
    """
    def __init__(self, speed=1):
        """
        Parameters
        ----------
        speed
            General tempo/ feel of Nebula's response (0.5 ~ moderate fast,
            1 ~ moderato, 2 ~ presto).
        """
        print('Building engine server')
        Listener.__init__(self)

        # Set global vars
        self.hivemind.running = True

        # Build the AI factory and pass it the data dict
        AIFactoryRAMI.__init__(self, speed)
        # self.BRAINBIT_CONNECTED = config.eeg_live
        self.BITALINO_CONNECTED = config.eda_live

        # Init brainbit reader
        # if self.BRAINBIT_CONNECTED:
        #     logging.info("Starting EEG connection")
        #     self.eeg_board = BrainbitReader()
        #     self.eeg_board.start()
        #     first_brain_data = self.eeg_board.read(1)
        #     logging.info(f'Data from brainbit = {first_brain_data}')

        # Init bitalino
        if self.BITALINO_CONNECTED:
            BITALINO_MAC_ADDRESS = config.mac_address
            BITALINO_BAUDRATE = config.baudrate
            BITALINO_ACQ_CHANNELS = config.channels

            eda_started = False
            while not eda_started:
                try:
                    self.eda = BITalino(BITALINO_MAC_ADDRESS)
                    eda_started = True
                except OSError:
                    print("Unable to connect to Bitalino")
                    retry = input("Retry (y/N)? ")
                    if retry.lower() != "y" and retry.lower() != "yes":
                        eda_started = True

            self.eda.start(BITALINO_BAUDRATE, BITALINO_ACQ_CHANNELS)
            first_eda_data = self.eda.read(1)[0, 1, 2, 3]
            logging.info(f'Data from BITalino = {first_eda_data}')

            # extent params for XYZ
            self.dancer_x_extents = config.dancer_x_extents
            self.dancer_y_extents = config.dancer_y_extents
            self.dancer_z_extents = config.dancer_z_extents

        # Work out master timing then collapse hivemind.running
        self.endtime = None

    def main_loop(self):
        """
        Starts the server / AI threads and gets the data rolling.
        """
        print('Starting the Nebula director')
        # Declare all threads
        t1 = Thread(target=self.make_data)
        t2 = Thread(target=self.snd_listen)
        t3 = Thread(target=self.dancer_input)

        # Start them all
        t1.start()
        t2.start()
        t3.start()

    def dancer_input(self):
        """
        Listen to live human input.
        """
        while self.hivemind.running:
            if time() >= self.endtime:
                break
            # Read data from bitalino
            if self.BITALINO_CONNECTED:

                #######################
                #  1. from EDA
                #######################

                # Get raw data from EDA
                eda_raw = [self.eda.read(1)[0][-4]]
                logging.debug(f"eda data raw = {eda_raw}")

                # Update raw EDA buffer
                eda_2d = np.array(eda_raw)[:, np.newaxis]
                self.hivemind.eda_buffer_raw = np.append(
                    self.hivemind.eda_buffer_raw, eda_2d, axis=1)
                self.hivemind.eda_buffer_raw = np.delete(
                    self.hivemind.eda_buffer_raw, 0, axis=1)

                # Detrend on the buffer time window
                eda_detrend = signal.detrend(self.hivemind.eda_buffer_raw)

                # Get min and max from raw EDA buffer
                eda_mins = np.min(eda_detrend, axis=1)
                eda_maxs = np.max(eda_detrend, axis=1)
                eda_mins = eda_mins - 0.05 * (eda_maxs - eda_mins)

                # Rescale between 0 and 1
                eda_norm = scaler(eda_detrend[:, -1], eda_mins, eda_maxs)

                # Update normalised EDA buffer
                eda_2d = eda_norm[:, np.newaxis]
                self.hivemind.eda_buffer = np.append(self.hivemind.eda_buffer,
                                                     eda_2d, axis=1)
                self.hivemind.eda_buffer = np.delete(self.hivemind.eda_buffer,
                                                     0, axis=1)

                #######################
                #  2. from XYZ
                #######################

                # Get raw data from XYZ (taken from robot arm code as uses same NNet in the AI Factory)
                dancer_xyz_raw =  [self.eda.read(1)[0][-1:-4]]
                norm_x = ((dancer_xyz_raw[0] - self.dancer_x_extents[0]) / (self.dancer_x_extents[1] - self.dancer_x_extents[0])) * (1 - 0) + 0
                norm_y = ((dancer_xyz_raw[1] - self.dancer_y_extents[0]) / (self.dancer_y_extents[1] - self.dancer_y_extents[0])) * (1 - 0) + 0
                norm_z = ((dancer_xyz_raw[2] - self.dancer_z_extents[0]) / (self.dancer_z_extents[1] - self.dancer_z_extents[0])) * (1 - 0) + 0

                norm_xyz = (norm_x, norm_y, norm_z)
                norm_xyz = tuple(np.clip(norm_xyz, 0.0, 1.0))
                norm_xy_2d = np.array(norm_xyz[:2])[:, np.newaxis]

                self.hivemind.current_dancer_x_y_z = norm_xyz
                self.hivemind.current_dancer_x_y = np.append(self.hivemind.current_dancer_x_y, norm_xy_2d, axis=1)
                self.hivemind.current_dancer_x_y = np.delete(self.hivemind.current_dancer_x_y, 0, axis=1)

                # add random X, Y, or Z for direct live stream
                self.hivemind.current_dancer_rnd = choice(norm_xyz)


            else:
                # Random data if no bitalino
                # Random data for EDA
                self.hivemind.eda_buffer = np.random.uniform(size=(1, 50))

                # Random data for XYZ
                rnd_xyz = (random(), random(), random())
                norm_xy_2d = np.array(rnd_xyz[:2])[:, np.newaxis]

                self.hivemind.current_dancer_x_y = np.append(self.hivemind.current_dancer_x_y, norm_xy_2d, axis=1)
                self.hivemind.current_dancer_x_y = np.delete(self.hivemind.current_dancer_x_y, 0, axis=1)

            sleep(0.1)  # for 10 Hz

        self.hivemind.running = False

    def terminate(self):
        """
        Terminate threads and connections like a grownup.
        """
        if self.BITALINO_CONNECTED:
            self.eda.close()
