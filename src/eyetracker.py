#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                        SCRIPT: eyetracker.py
#
#
#                   DESCRIPTION: Eye tracker interface and utilities
#
#
#                          RULE: DAYW
#
#
#
#                       CREATOR: Sharif Saleki
#                          TIME: 09-23-2025-7810598105114117
#                         SPACE: Stanford Univeristy, Stanford, CA
#
# =================================================================================================== #
from pathlib import Path
import os

import numpy as np
import pandas as pd

import pylink
from psychopy import core, logging


class MyeLink:
    """
    Eyelink interface class
    """
    def __init__(self, config, debug=False):
        self.debug = debug
        self.config = config
        self.tracker = None

    def connect(self):
        if self.debug:
            self.tracker = pylink.EyeLink(None)
        else:
            try:
                self.tracker = pylink.EyeLink(self.config["address"])
            except RuntimeError as error:
                logging.log(level=logging.ERROR, msg=f"Failed to connect to EyeLink Tracker: {error}")
                core.quit()

    def open_data_file(self, edf_filename):
        """
        Open an EDF data file on the Host PC

        Parameters
        ----------
        edf_filename : str
            EDF file name (8 characters max, no extension)

        Returns
        -------
        None
        """
        edf_filename = str(edf_filename)[:8]  # EDF file name must be 8 characters or less
        try:
            self.tracker.openDataFile(edf_filename)
        except RuntimeError as err:
            logging.log(level=logging.ERROR, msg=f"Failed to open EDF file on Host PC: {err}")
            # close the link if we have one open
            if self.tracker.isConnected():
                self.tracker.close()
            core.quit()

    def add_preamble_text(self, text):
        """
        Add preamble text to the EDF file

        Parameters
        ----------
        text : str
            text to be added

        Returns
        -------
        None
        """
        try:
            self.tracker.sendCommand(f'add_file_preamble_text "{text}"')
        except RuntimeError as e:
            logging.log(level=logging.ERROR, msg=f"Failed to add preamble text to EDF file: {e}")

    def offline(self):
        """
        Put the tracker in offline mode

        Returns
        -------
        None
        """
        self.tracker.setOfflineMode()

    def check_version(self):
        """
        Get the EyeLink tracker software version

        Returns
        -------
        int
            software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
            5-EyeLink 1000 Plus, 6-Portable DUO
        """
        eyelink_ver = 0  # set version to 0, in case running in Dummy mode
        if not self.debug:
            vstr = self.tracker.getTrackerVersionString()
            eyelink_ver = int(vstr.split()[-1].split('.')[0])
            # print out some version info in the shell
            logging.log(level=logging.INFO, msg=f'Running experiment on {vstr}, version {eyelink_ver}')
        return eyelink_ver

    def setup_data_filters(self):
        """
        Set up the data filters for the EyeLink tracker

        Returns
        -------
        None
        """
        # set the file and link data filters
        self.tracker.sendCommand(f"sample_rate {self.config['sample_rate']}")
        self.tracker.sendCommand(f"file_event_filter = {self.config['file_event_filter']}")
        self.tracker.sendCommand(f"file_sample_data = {self.config['file_sample_data']}")
        self.tracker.sendCommand(f"link_event_filter = {self.config['link_event_filter']}")
        self.tracker.sendCommand(f"link_sample_data = {self.config['link_sample_data']}")

    def setup_calibration(self, window):
        """
        Set up the EyeLink tracker calibration

        Parameters
        ----------
        window : psychopy.visual.Window
            PsychoPy window

        Returns
        -------
        None
        """
        self.tracker.sendCommand(f"calibration_type = {self.config['calibration_type']}")

        # Perform the calibration
        genv = pylink.EyeLinkCoreGraphicsPsychoPy(window)
        self.tracker.doTrackerSetup(genv)
