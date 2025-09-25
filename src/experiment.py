#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                     SCRIPT: experiment.py
#
#
#                DESCRIPTION: Script for experiment functions
#
#
#                       RULE: DAYW
#
#
#
#                    CREATOR: Sharif Saleki
#                       TIME: 09-22-2025-7810598105114117
#                      SPACE: Stanford Univeristy, Stanford, CA
#
# =================================================================================================== #
from psychopy import visual, event


def clear_screen(window, color=[0, 0, 0]):
    """
    clear up the PsychoPy window

    Parameters
    ----------
    window : psychopy.visual.Window
        PsychoPy window
    color : list, optional
        background color, by default [0, 0, 0]

    Returns
    -------
    None
    """
    window.fillColor = color
    window.flip()


def show_msg(window, text, color=[-1, -1, -1], wait_for_keypress=True):
    """
    Show task instructions on screen

    Parameters
    ----------
    window : psychopy.visual.Window
        PsychoPy window
    text : str
        text to be displayed
    color : list, optional
        text color, by default [-1, -1, -1]
    wait_for_keypress : bool, optional
        whether to wait for a key press to continue, by default True

    Returns
    -------
    None
    """
    msg = visual.TextStim(
        window,
        text,
        color=color,
        wrapWidth=window.size[0] * 0.8,
    )
    clear_screen(window)
    msg.draw()
    window.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(window)


def make_display()
