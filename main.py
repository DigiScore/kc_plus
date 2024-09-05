import art
import logging
import time

import config
from modules.conducter import Conducter
from nebula.hivemind import DataBorg
from nebula.nebula import Nebula

if config.data_writer:
    from modules.data_writer import DataWriter


class Main:
    """
    Main script to start the robot arm drawing digital score work.
    Conducter calls the local interpreter for project specific functions. This
    communicates directly to the robot libraries.
    Nebula kick-starts the AI Factory for generating NNet data and affect
    flows.
    This script also controls the live mic audio analyser.
    Paramaters are to be modified in config.py.
    """
    def __init__(self):
        art.tprint("KC Plus")
        # Build initial dataclass filled with random numbers
        self.hivemind = DataBorg()

        # Logging for all modules
        logging.basicConfig(level=logging.INFO)

        # Init the AI factory (inherits AIFactory, Listener)
        nebula = Nebula(speed=config.speed)

        # Init Conducter & Gesture management (controls XArm)
        robot = Conducter(speed=config.speed)

        # Init data writer
        if config.data_writer:
            dw = DataWriter()

        # Start Nebula AI Factory after conducter starts data moving
        nebula.endtime = time.time() + config.duration_of_piece
        self.hivemind.running = True
        robot.main_loop()
        nebula.main_loop()
        if config.data_writer:
            dw.main_loop()

if __name__ == "__main__":
    Main()
