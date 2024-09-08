import logging
from random import random, randrange, uniform
from threading import Thread
from time import sleep, time

import config
from nebula.hivemind import DataBorg


class Conducter:
    """
    Controls movement and shapes drawn by the robot.
    """
    def __init__(self, speed: int = 5):

        self.XARM_CONNECTED = config.xarm_connected

        if self.XARM_CONNECTED:
            from modules.draw_xarm import Drawbot

            port = config.xarm1_port
            self.drawbot = Drawbot(port)

        else:
            self.drawbot = None

        # Own the dataclass
        self.hivemind = DataBorg()

        # Start operating vars
        self.current_phrase_num = 0  # number of phrases looped through, can be used to change behaviour over time...
        self.joint_inc = 10          # scaling factor for incremental movement
        self.continuous_mode = 0     # mode for continuous module. 0 == on page, 1 == above page
        self.continuous_source = 0   # source of data used for continous movement. 0 == random, 1 == NN, 2 == peak
        self.global_speed = speed
        self.mic_in_prediction = config.mic_in_prediction
        self.phrase_length_extents_min = config.phrase_length_extents[0]
        self.phrase_length_extents_max = config.phrase_length_extents[1]
        self.rhythm_length_extents_min = config.rhythm_length_extents[0]
        self.rhythm_length_extents_max = config.rhythm_length_extents[1]
        self.arm_speed_min = config.arm_speed_extents[0]
        self.arm_speed_max = config.arm_speed_extents[1]

        # Get the baseline temperature from config
        self.temperature = config.temperature

        if self.drawbot:
            # self.drawbot.home()
            self.drawbot.go_position_draw()

            input('To start press ENTER')
            # print('Going to draw position...')
            # input('Adjust pen height, then press ENTER')
            # self.drawbot.go_position_one_two()
            self.drawbot.go_position_ready()

    def main_loop(self):
        """
        Starts the main thread for the gesture manager
        """
        gesture_thread = Thread(target=self.gesture_manager)
        gesture_thread.start()

        if self.drawbot:
            position_thread = Thread(target=self.drawbot.get_normalised_position)
            position_thread.start()
            self.drawbot.command_list_main_loop()

    def gesture_manager(self):
        """
        Listens to the realtime incoming signal and calculates an affectual
        response based on general boundaries:
            HIGH   - if input stream is LOUD (0.8+) then emit, smash a random
                     fill and break out to Daddy cycle.
            MEDIUM - if input energy is 0.1-0.8 then emit, a jump out of child
                     loop.
            LOW    - nothing happens, continues with cycles.
        """

        if self.drawbot:
            print("Started robot control thread")
            self.drawbot.go_position_draw()

        # Names for affect listening
        stream_list = config.stream_list
        stream_list_len = len(stream_list)

        while self.hivemind.running:
            ###################################################################
            # Phrase-level gesture gate: 3 - 8 seconds
            ###################################################################
            if self.XARM_CONNECTED:
                self.drawbot.random_pen()

            # Flag for breaking a phrase from big affect signal
            self.hivemind.interrupted = False

            # Clear command list at start of each gesture cycle
            if self.drawbot:
                self.drawbot.clear_commands()

            # Get length of gesture
            phrase_length = (randrange(self.phrase_length_extents_min, self.phrase_length_extents_max) / 100)  # + self.global_speed
            phrase_loop_end = time() + phrase_length

            print(f"======== GESTURE - Daddy cycle started ========", end=' ')
            print(f"Duration =  {phrase_length} seconds")

            ###################################################################
            # Randomly pick an input stream for this cycle
            # (either mic_in or stream in config.stream_list)
            ###################################################################
            if random() < self.mic_in_prediction:
                if random() < 0.5:
                    rnd_stream = 'mic_in'
                else:
                    rnd_stream = 'current_dancer_xyz_rnd'
            else:
                rnd = randrange(stream_list_len)
                rnd_stream = stream_list[rnd]

            self.hivemind.thought_train_stream = rnd_stream
            print(f"Random stream = {self.hivemind.thought_train_stream}")

            while time() < phrase_loop_end:
                ###############################################################
                # Rhythm-level gesture gate: .5-2 seconds
                ###############################################################

                # if a major break out then go to Daddy cycle and restart
                if self.hivemind.interrupted:
                    print("----------------STREAM INTERRUPT----------------")
                    break

                # Clear the alarms
                if self.drawbot:
                    self.drawbot.clear_alarms()

                # Generate rhythm rate here
                rhythm_loop_end_time = time() + (randrange(self.rhythm_length_extents_min, self.rhythm_length_extents_max) / 1000)
                logging.debug(f'end time = {rhythm_loop_end_time}')

                # Speed for this phrase
                arm_speed = randrange(self.arm_speed_min, self.arm_speed_max)
                if self.XARM_CONNECTED:
                    self.drawbot.set_speed(arm_speed)

                while time() < rhythm_loop_end_time:
                    ###########################################################
                    # Stream the chosen data around a loop
                    ###########################################################

                    # Make master output the current value of affect stream
                    # 1. Go get the current value from dict
                    thought_train = getattr(self.hivemind, rnd_stream)
                    logging.debug(f'======== RHYTHM cycle ========'
                                  f'|Affect stream output {rnd_stream}'
                                  f' == {thought_train}')

                    # 2. Send to Master Output
                    self.hivemind.master_stream = thought_train

                    ###########################################################
                    # Makes a response to chosen thought stream
                    ###########################################################
                    # [HIGH response]
                    if thought_train > 0.8 or self.hivemind.interrupted:
                        print('Interrupt > !!! HIGH !!!')

                        # A - Refill dict with random
                        self.hivemind.randomiser()

                        # B - Jumps out of this loop into daddy
                        # (will clear commands by detecting interupt_clear)
                        self.hivemind.interrupted = True

                        # C - Respond
                        if self.drawbot:
                            self.high_energy_response()

                        # D- Break out of this loop, and next because of flag
                        sleep(0.1)
                        break

                    # [LOW response]
                    elif thought_train < 0.1:
                        print('Interrupt < LOW : no response')
                        if self.drawbot:
                            if random() < 0.36:
                                self.design_move(thought_train)

                    # [MEDIUM response]
                    else:
                        if self.drawbot:
                            self.design_move(thought_train)

                    sleep(0.1)

        logging.info('quitting director thread')
        self.terminate()


    def design_move(self, thought_train):
        """
        Takes a thought train value and maps it to one of 10 functions, salvaged from Jess+ belief module.
        Due to the high energy interrupt in gesture manager this range is: 0.0 - 0.8
        """

        # get current position
        x, y = self.drawbot.get_pose()[:2]
        arc_range = thought_train * 10

        # make a random choice (for now)
        # todo - maybe map these across the range of input thought-trains
        randchoice = randrange(13)
        logging.debug(f'Random choice: {randchoice}')

        match randchoice:
            case 0:
                decision_type = 'draw line'
                self.drawbot.go_draw(x + self.rnd(thought_train*10),
                                     y + self.rnd(thought_train*10),
                                     False)

            case 1:
                decision_type = 'random character'
                self.drawbot.draw_random_char(thought_train * randrange(10, 20))

            case 2:
                decision_type = 'dot'
                self.drawbot.dot()

            case 3:
                decision_type = 'note head'
                note_size = randrange(1, 10)
                self.drawbot.note_head(size=note_size)

            case 4:
                decision_type = 'note head and line'
                note_size = randrange(1, 10)
                self.drawbot.note_head(size=note_size)
                self.drawbot.position_move_by(self.rnd(thought_train*10),
                                              self.rnd(thought_train*10),
                                              0, wait=True)

            case 5:
                decision_type = 'random jump'
                self.drawbot.go_random_jump()

            case 6:
                decision_type = 'draw arc'
                self.drawbot.arc2D(x + self.rnd(arc_range),
                                   y + self.rnd(arc_range),
                                   x + self.rnd(arc_range),
                                   y + self.rnd(arc_range))

            case 7:
                decision_type = 'small squiggle'
                squiggle_list = []
                for _ in range(randrange(3, 9)):
                    squiggle_list.append((self.rnd(arc_range),
                                          self.rnd(arc_range),
                                          self.rnd(arc_range)))
                self.drawbot.squiggle(squiggle_list)

            case 8:
                decision_type = 'draw circle'
                side = randrange(2)
                self.drawbot.draw_circle(int(arc_range), side)

            case 9:
                decision_type = 'arc'
                self.drawbot.go_draw(x + self.rnd(arc_range),
                                     y + self.rnd(arc_range))

            case 10:
                decision_type = 'return to coord'
                self.drawbot.return_to_coord()

            case 11:
                decision_type = 'random shape group'
                self.drawbot.create_shape_group()
                # self.drawbot.repeat_shape_group()

            case 12:
                decision_type = 'random pen move'
                self.drawbot.random_pen()

        # log the decision
        logging.info(decision_type)
        self.hivemind.design_decision = decision_type


    def high_energy_response(self):
        """
        Clear commands to interupt current gesture and moves on to new ones.
        """
        # self.drawbot.clear_commands()
        # self.drawbot.go_random_jump()
        high_energy_log = "Drawing High Energy random 3D"
        logging.info(high_energy_log)
        self.hivemind.design_decision = high_energy_log
        if self.XARM_CONNECTED:
            self.drawbot.go_random_3d()

    def terminate(self):
        """
        Smart collapse of all threads and comms.
        """
        print('TERMINATING')
        if self.drawbot:
            self.drawbot.go_position_ready()
            self.drawbot.go_position_one_two()
            self.drawbot.home()
            self.drawbot.clear_commands()
            # if self.DOBOT_CONNECTED:
            #     self.drawbot.close()
            if self.XARM_CONNECTED:
                self.drawbot.set_fence_mode(False)
                self.drawbot.disconnect()

    def rnd(self, power_of_command: int) -> int:
        """
        Returns a randomly generated positive or negative integer, influenced
        by the incoming power factor.
        """
        power_of_command = int(power_of_command)
        if power_of_command <= 0:
            power_of_command = 1
        pos = 1
        if random() >= 0.5:
            pos = -1
        result = (randrange(1, 5) + randrange(power_of_command)) * pos
        if result == 0:
            result = 1
        logging.debug(f'Rnd result = {result}')
        return result
