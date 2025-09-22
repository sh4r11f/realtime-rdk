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
    """ clear up the PsychoPy window"""
    window.fillColor = color
    window.flip()


def show_msg(window, text, color=[-1, -1, -1], wait_for_keypress=True):
    """ Show task instructions on screen"""

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
