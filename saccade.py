#
# Copyright (c) 1996-2021, SR Research Ltd., All Rights Reserved
#
# For use by SR Research licencees only. Redistribution and use in source
# and binary forms, with or without modification, are NOT permitted.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the distribution.
#
# Neither name of SR Research Ltd nor the name of contributors may be used
# to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS
# IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# DESCRIPTION:
# This example scripts shows how retrieve to eye events (saccades) during
# testing. A visual target appears on the leftside or right side of the
# screen and the participant is required to quickly shift gaze to look
# at the target (pro-saccade) or a mirror location on the opposite side
# of the central fixation (anti-saccade).

# Last updated: 3/29/2021
from __future__ import division
from __future__ import print_function

from pathlib import Path

import os
import platform
from datetime import date
import time
import sys
import yaml

import numpy as np
import pandas as pd

import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, logging, gui
from psychopy.tools.monitorunittools import deg2pix

from math import fabs, hypot
import nidaqmx
from nidaqmx.constants import VoltageUnits

# Switch to the script folder
root = Path().cwd()
if root.is_dir():
    os.chdir(root)

# Directories
res_dir = root / 'resources'
src_dir = root / 'src'
data_dir = root / 'data'
config_dir = root / 'config'

# Show only critical log message in the PsychoPy console
logging.console.setLevel(logging.CRITICAL)

# Set this variable to True if you use the built-in retina screen as your
# primary display device on macOS. If have an external monitor, set this
# variable True if you choose to "Optimize for Built-in Retina Display"
# in the Displays preference settings.
use_retina = False

# Set this variable to True to run the script in "Dummy Mode"
dummy_mode = False

# Set this variable to True to run the task in full screen mode
# It is easier to debug the script in non-fullscreen mode
full_screen = True
TRIALS_PER_BLOCK = 12
NUM_BLOCKS = 3

# Store the parameters of all trials in a list, here we block the
# anti- and pro-saccade trials
# [cond, tar_pos, correct_sac_tar_pos]
block_types = ['go', 'no-go', 'noise']

# use a dictionary to label target position (left vs. right)
tar_pos = (-350, 0)

# Set up EDF data file name and local data folder
#
# The EDF data filename should not exceed 8 alphanumeric characters
# use ONLY number 0-9, letters, & _ (underscore) in the filename
edf_fname = 'rdk'

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
dlg_title = 'Session'
dlg_prompt = 'Please enter session info ;)'
dlg = gui.Dlg(dlg_title)
dlg.addText(dlg_prompt)
dlg.addField("Subject", "AQ")
dlg.addField('Session', '')

# show dialog and wait for OK or Cancel
ok_data = dlg.show()
if dlg.OK:  # if ok_data is not None
    sub_id = ok_data[0]
    ses_id = int(ok_data[1])
    print('Subject: {}'.format(sub_id))
    print('Session: {}'.format(ses_id))
else:
    print('user cancelled')
    core.quit()
    sys.exit()

# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
root = Path().cwd()
today = date.today().strftime("%Y_%m_%d")
data_dir = root / 'data' / today / f"sub-{sub_id}" / f"ses-{ses_id:02d}"
data_dir.mkdir(exist_ok=True, parents=True)
local_edf = str(data_dir / f"sub-{sub_id}_ses-{ses_id}_rdk_{today}.edf")

# Step 1: Connect to the EyeLink Host PC
#
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"
if dummy_mode:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()

# Step 2: Open an EDF data file on the Host PC
edf_file = "rdk.EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add a header text to the EDF file to identify the current experiment name
# This is OPTIONAL. If your text starts with "RECORDED BY " it will be
# available in DataViewer's Inspector window by clicking
# the EDF session node in the top panel and looking for the "Recorded By:"
# field in the bottom panel of the Inspector.
# preamble_text = 'RECORDED BY DX' % Path(__file__).name
# el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)
try:
    el_tracker.sendCommand("add_file_preamble_text 'RECORDED BY DX'")
except RuntimeError as e:
    print('ERROR:', e)

# Step 3: Configure the tracker
#
# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
# 5-EyeLink 1000 Plus, 6-Portable DUO
eyelink_ver = 0  # set version to 0, in case running in Dummy mode
if not dummy_mode:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    # print out some version info in the shell
    print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

# File and Link data control
# what eye events to save in the EDF file, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
# what eye events to make available over the link, include everything by default
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# what sample data to save in the EDF data file and to make available
# over the link, include the 'HTARGET' flag to save head target sticker
# data for supported eye trackers
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Optional tracking parameters
# Sample rate, 250, 500, 1000, or 2000, check your tracker specification
# if eyelink_ver > 2:
#     el_tracker.sendCommand("sample_rate 1000")
# Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
el_tracker.sendCommand("calibration_type = HV5")
# Set a gamepad button to accept calibration/drift check target
# You need a supported gamepad/button box that is connected to the Host PC
el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

# Step 4: set up a graphics environment for calibration
#
# Open a window, be sure to specify monitor parameters
screen_num = 1

mon = monitors.Monitor('myMonitor', width=52.0, distance=42.0)
mon.setSizePix((1920, 1080))
win = visual.Window(fullscr=full_screen,
                    screen=screen_num,
                    monitor=mon,
                    winType='pyglet',
                    units='pix')

# get the native screen resolution used by PsychoPy
scn_width, scn_height = win.size
# resolution fix for Mac retina displays
if 'Darwin' in platform.system():
    if use_retina:
        scn_width = int(scn_width/2.0)
        scn_height = int(scn_height/2.0)

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
# see the EyeLink Installation Guide, "Customizing Screen Settings"
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization, see Data
# Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
print(genv)  # print out the version number of the CoreGraphics library

# register a reward callback so each accepted calibration target can trigger reward
# try:
#     genv.set_reward_callback(lambda: give_reward())
# except Exception:
#     pass

# Set background and foreground colors for the calibration target
# in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
foreground_color = (-1, -1, -1)
background_color = win.color
genv.setCalibrationColors(foreground_color, background_color)

# Set up the calibration target
#
# The target could be a "circle" (default), a "picture", a "movie" clip,
# or a rotating "spiral". To configure the type of calibration target, set
# genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
# genv.setTargetType('picture')
#
# Use gen.setPictureTarget() to set a "picture" target
# genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))
#
# Use genv.setMovieTarget() to set a "movie" target
# genv.setMovieTarget(os.path.join('videos', 'calibVid.mov'))

# Use the default calibration target ('circle')
genv.setTargetType('picture')

res_dir = Path(__file__).parent / "resources" / "calibration"
calib_files = list(res_dir.glob("*.png"))
calib_images = list(np.random.choice(calib_files, 5))
genv.setPictureTarget(calib_images)

# Configure the size of the calibration target (in pixels)
# this option applies only to "circle" and "spiral" targets
genv.setTargetSize(100)

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('qbeep.wav', 'type.wav', 'error.wav')

# resolution fix for macOS retina display issues
if use_retina:
    genv.fixMacRetinaDisplay()

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)

# define a few helper functions for trial handling


def clear_screen(win):
    """ clear up the PsychoPy window"""

    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_msg(win, text, wait_for_keypress=True):
    """ Show task instructions on screen"""

    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def give_reward(reward_voltage=5.0, duration_ms=200, pulses=2, inter_pulse_ms=200):
    """
    Hardware‑timed reward output matching the MATLAB session approach.
    Builds a waveform sampled at 1000 Hz and outputs it using a finite
    buffered AO task for more precise timing than software sleep.
    """
    device_name = 'Dev1'  # update to your device name if different
    channel_id = 'ao0'
    sample_rate = 1000  # Hz (matches MATLAB session Rate)

    # enforce sane bounds
    try:
        pulses = int(pulses)
    except Exception:
        pulses = 1
    pulses = max(1, min(pulses, 100))

    duration_ms = max(1, int(duration_ms))
    inter_pulse_ms = max(0, int(inter_pulse_ms))

    on_samples = max(1, int(round(duration_ms * sample_rate / 1000.0)))
    off_samples = int(round(inter_pulse_ms * sample_rate / 1000.0))

    # build one pulse (on then off) and tile it
    single = np.concatenate([np.ones(on_samples) * float(reward_voltage),
                                np.zeros(off_samples)])
    signal = np.tile(single, pulses)

    # fallback: ensure signal not empty
    if signal.size == 0:
        signal = np.array([float(reward_voltage)])

    try:
        from nidaqmx.constants import AcquisitionType
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(
                f"{device_name}/{channel_id}",
                min_val=0.0, max_val=10.0, units=VoltageUnits.VOLTS
            )
            task.timing.cfg_samp_clk_timing(rate=sample_rate,
                                            sample_mode=AcquisitionType.FINITE,
                                            samps_per_chan=len(signal))
            # write and start the finite buffered output
            task.write(signal.tolist(), auto_start=True)
            task.wait_until_done(timeout=10.0 + len(signal) / sample_rate)
            task.stop()
    except Exception as e:
        print("Warning: failed to give hardware‑timed reward:", e)

        print(f"Reward (buffered): {reward_voltage}V x{pulses} pulses, {duration_ms}ms on, {inter_pulse_ms}ms gap")


# def give_reward(reward_voltage=5.0, duration_ms=200, pulses=2, inter_pulse_ms=100):
#     """
#     Deliver one or more reward pulses on the analog output.

#     Parameters:
#     - reward_voltage: voltage for each pulse (0-10V)
#     - duration_ms: duration of each pulse in milliseconds
#     - pulses: number of pulses to deliver (integer >=1)
#     - inter_pulse_ms: gap between pulses in milliseconds (voltage 0 during gap)
#     """
#     device_name = 'Dev1'  # update as appropriate
#     channel_id = 'ao0'

#     # sanity limits
#     try:
#         pulses = int(pulses)
#     except Exception:
#         pulses = 1
#     pulses = max(1, min(pulses, 20))  # cap to 20 pulses

#     inter_pulse_ms = max(0, int(inter_pulse_ms))

#     try:
#         with nidaqmx.Task() as task:
#             task.ao_channels.add_ao_voltage_chan(
#                 f"{device_name}/{channel_id}",
#                 min_val=0.0, max_val=10.0, units=VoltageUnits.VOLTS
#             )
#             for i in range(pulses):
#                 # Write the reward voltage, hold for duration, then write 0
#                 task.write(float(reward_voltage))
#                 time.sleep(duration_ms / 1000.0)
#                 task.write(0.0)
#                 if i < (pulses - 1) and inter_pulse_ms > 0:
#                     time.sleep(inter_pulse_ms / 1000.0)
#     except Exception as e:
#         print("Warning: failed to give reward:", e)

#     # small print for debugging/logging
#     print(f"Reward: {reward_voltage}V x{pulses} pulses, {duration_ms}ms each, {inter_pulse_ms}ms gap")


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        # local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()


    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial():
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR


def acquire_fixation(el_tracker, timeout_s=3.0, hold_time_s=0.05, blink_interval_s=0.1, fix_radius_px=5):
    """Blink the fixation spot until fixation is acquired or timeout.

    The fixation stimulus is drawn in red and uses the provided radius
    (pixels). Returns True if fixation acquired, False on timeout.
    """
    # center coords
    cx = scn_width / 2.0
    cy = scn_height / 2.0
    # tolerance for fixation (pixels)
    fix_tol_px = deg2pix(2.0, mon)

    start_t = time.time()
    fix_visible = True
    last_blink = start_t
    hold_start = None

    # PsychoPy mouse fallback for simulation/dummy
    mouse = event.Mouse(win=win, visible=True)

    while (time.time() - start_t) < timeout_s:
        now = time.time()
        # toggle blink state when interval elapsed
        if (now - last_blink) >= blink_interval_s:
            fix_visible = not fix_visible
            last_blink = now

        # draw fixation if visible
        clear_screen(win)
        if fix_visible:
            fix_dot = visual.Circle(
                win,
                radius=fix_radius_px,
                fillColor='red',
                lineColor='red',
                pos=(0, 0),
            )
            fix_dot.draw()
        win.flip()

        # get newest EyeLink sample if possible
        try:
            sample = el_tracker.getNewestSample()
        except Exception:
            sample = None

        gx = gy = None
        gaze_center_x = gaze_center_y = None

        # Extract gaze from EyeLink sample (screen coords) when available
        if sample is not None:
            try:
                gaze = sample.getLeftEye().getGaze()
                if gaze is not None:
                    gx, gy = gaze
            except Exception:
                try:
                    gx = sample.gx
                    gy = sample.gy
                except Exception:
                    gx = gy = None

        if (gx is not None) and (gy is not None):
            # convert absolute screen coords to center-based coords
            gaze_center_x = gx - cx
            gaze_center_y = gy - cy
        else:
            # fallback to PsychoPy mouse (already center-based)
            try:
                mx, my = mouse.getPos()
                gaze_center_x = mx
                gaze_center_y = my
            except Exception:
                gaze_center_x = gaze_center_y = None

        # if we have gaze_center coords, check distance to center
        if (gaze_center_x is not None) and (gaze_center_y is not None):
            dx = gaze_center_x
            dy = gaze_center_y
            if (dx * dx + dy * dy) <= (fix_tol_px * fix_tol_px):
                if hold_start is None:
                    hold_start = now
                elif (now - hold_start) >= hold_time_s:
                    clear_screen(win)
                    return True
            else:
                hold_start = None

        # small delay to prevent tight spin
        pylink.pumpDelay(10)

    # timeout expired
    clear_screen(win)
    return False


def run_trial(cond, trial_index, stim_params, task_params):
    """Run a single RDK trial.

    cond - one of 'go', 'no-go', or 'noise'
    trial_index - order index

    Returns acc (1=correct, 0=incorrect)
    """

    # RDK display parameters (position relative to center)
    rdk_params = stim_params["RDK"]
    fix_params = stim_params["Fixation"]

    rdk_ecc = deg2pix(rdk_params["eccentricity"], mon)
    rdk_pos_jitter = deg2pix(rdk_params["position_jitter"], mon)
    # rdk_nDots and rdk_speed are not used in this script; keep calculations
    # minimal to avoid unused-variable linter warnings by not assigning them.
    rdk_dur = rdk_params["duration"]
    rdk_dotSize = deg2pix(rdk_params["dot_size"], mon)
    # prepare the central fixation cross
    fix_cross = visual.Circle(
        win,
        radius=deg2pix(fix_params["size"], mon),
        fillColor=fix_params["color"],
        lineColor=fix_params["color"]
    )

    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()

    # Acquire fixation before stimulus: blink fixation spot until fixation
    # is acquired or timeout. If fixation not acquired, mark trial failed.
    # fix_ok = acquire_fixation(
    #     el_tracker,
    #     timeout_s=2.0,
    #     blink_interval_s=0.5,
    #     fix_radius_px=deg2pix(fix_params["size"], mon),
    # )
    fix_ok = True
    if not fix_ok:
        el_tracker.sendMessage('fixation_not_acquired')
        # ensure not recording
        if el_tracker.isRecording():
            pylink.pumpDelay(50)
            el_tracker.stopRecording()
        # return failed trial
        return 0

    # draw cross on the Host PC display to help visualization
    cross_coords = (int(scn_width/2.0), int(scn_height/2.0))
    el_tracker.sendCommand('clear_screen 0')  # clear the host Display
    el_tracker.sendCommand('draw_cross %d %d 10' % cross_coords)

    # send a "TRIALID" message to mark the start of a trial
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIALID %d' % trial_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'TRIAL number %d, %s' % (trial_index, cond)
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # drift check
    # we recommend drift-check at the beginning of each trial
    # the doDriftCorrect() function requires target position in integers
    # the last two arguments:
    # draw_target (1-default, 0-draw the target then call doDriftCorrect)
    # allow_setup (1-press ESCAPE to recalibrate, 0-not allowed)
    #
    # Skip drift-check if running the script in Dummy Mode
    # while not dummy_mode:
    #     # terminate the task if no longer connected to the tracker or
    #     # user pressed Ctrl-C to terminate the task
    #     if (not el_tracker.isConnected()) or el_tracker.breakPressed():
    #         terminate_task()
    #         return pylink.ABORT_EXPT

    #     # drift-check and re-do camera setup if ESCAPE is pressed
    #     try:
    #         error = el_tracker.doDriftCorrect(int(scn_width/2.0),
    #                                           int(scn_height/2.0), 1, 1)
    #         # break following a success drift-check
    #         if error is not pylink.ESC_KEY:
    #             break
    #     except Exception as e:
    #         print('Drift correction error:', e)

    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()

    # Start recording
    # arguments: sample_to_file, events_to_file, sample_over_link,
    # event_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial()
        return pylink.TRIAL_ERROR

    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)

    # determine which eye(s) is/are available
    # 0- left, 1-right, 2-binocular
    eye_used = el_tracker.eyeAvailable()
    if eye_used == 1:
        el_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        el_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("Error in getting the eye information!")
        return pylink.TRIAL_ERROR

    # put the central fixation cross on screen and attempt a simple acquire
    # fix_cross.draw()
    # win.flip()
    # send over a message to log the onset of the fixation cross
    # el_tracker.sendMessage('fix_onset')

    # OPTIONAL - draw fixation cross guide in Data Viewer
    h_center = int(scn_width/2.0)
    v_center = int(scn_height/2.0)
    line_hor = (h_center - 20, v_center, h_center + 20, v_center)
    line_ver = (h_center, v_center - 20, h_center, v_center + 20)
    el_tracker.sendMessage('!V CLEAR 128 128 128')  # clear the screen
    el_tracker.sendMessage('!V DRAWLINE 0 255 0 %d %d %d %d' % line_hor)
    el_tracker.sendMessage('!V DRAWLINE 0 255 0 %d %d %d %d' % line_ver)

    # Simple, non-blinking fixation acquire: check gaze samples for timeout
    fix_tol_px = deg2pix(task_params["fixation_tolerance"], mon)
    timeout_s = 2.0
    hold_time_s = task_params["fixation_hold"] / 1000.0
    start_t = time.time()
    hold_start = None
    mouse = event.Mouse(win=win, visible=True)
    fix_acquired = False
    fix_on = True
    blink_period = task_params["fixation_blink_period"] / 1000.0
    blink_start = time.time()
    cx = scn_width / 2.0
    cy = scn_height / 2.0

    while (time.time() - start_t) < timeout_s:
        # abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return 0

        # Draw blinking fixation
        if time.time() - blink_start > blink_period:
            fix_on = not fix_on
            blink_start = time.time()

        if fix_on:
            fix_cross.draw()
        win.flip()

        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # immediate quit
            if keycode == 'escape':
                el_tracker.sendMessage('trial_skipped_by_user')
                clear_screen(win)
                abort_trial()
                return 0
            # Ctrl-C to terminate experiment (existing behavior)
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task()
                return 0
            # Pause the experiment mid-trial and return control to the outer loop
            if keycode == 'p':
                el_tracker.sendMessage('paused_by_user')
                # stop recording cleanly
                if el_tracker.isRecording():
                    pylink.pumpDelay(100)
                    el_tracker.stopRecording()
                clear_screen(win)
                return None
            # Calibrate immediately mid-trial (non-Ctrl 'c')
            if keycode == 'c' and not modifier.get('ctrl'):
                el_tracker.sendMessage('calibration_requested')
                # stop recording cleanly before calibration
                if el_tracker.isRecording():
                    pylink.pumpDelay(100)
                    el_tracker.stopRecording()
                try:
                    el_tracker.doTrackerSetup()
                except RuntimeError as err:
                    print('Calibration error during trial:', err)
                    try:
                        el_tracker.exitCalibration()
                    except Exception:
                        pass
                # return to outer loop so the experimenter can choose next action
                clear_screen(win)
                return None
            # give reward
            if keycode == 'r':
                give_reward()
                print("Reward given")

        # try to get newest sample from EyeLink
        try:
            sample = el_tracker.getNewestSample()
        except Exception:
            sample = None

        gx = gy = None
        gaze_center_x = gaze_center_y = None

        if sample is not None:
            try:
                gaze = sample.getLeftEye().getGaze()
                if gaze is not None:
                    gx, gy = gaze
            except Exception:
                try:
                    gx = sample.gx
                    gy = sample.gy
                except Exception:
                    gx = gy = None

        if (gx is not None) and (gy is not None):
            gaze_center_x = gx - cx
            gaze_center_y = gy - cy
        else:
            try:
                mx, my = mouse.getPos()
                gaze_center_x = mx
                gaze_center_y = my
            except Exception:
                gaze_center_x = gaze_center_y = None

        if (gaze_center_x is not None) and (gaze_center_y is not None):
            dx = gaze_center_x
            dy = gaze_center_y
            if (dx * dx + dy * dy) <= (fix_tol_px * fix_tol_px):
                if hold_start is None:
                    hold_start = time.time()
                elif (time.time() - hold_start) >= hold_time_s:
                    fix_acquired = True
                    break
            else:
                hold_start = None

        pylink.pumpDelay(10)

    if not fix_acquired:
        # fixation failed: mark trial incorrect and skip
        el_tracker.sendMessage('fixation_not_acquired')
        if el_tracker.isRecording():
            pylink.pumpDelay(50)
            el_tracker.stopRecording()
        return 0

    # show the RDK depending on condition
    # build DotStim: direction in degrees (90 = up, 270 = down)
    if cond == 'go':
        coherence = 1.0
        direction = 90
    elif cond == 'no-go':
        coherence = 1.0
        direction = 270
    else:  # 'noise'
        coherence = 0.0
        direction = 0

    # Jitter location while keeping the eccentricity the same
    # side = np.random.choice([0, 1])
    side = 1 if stim_params["RDK"]["side"] == "right" else 0
    pos_jitter = np.random.uniform(-rdk_pos_jitter, rdk_pos_jitter)
    if side == 1:
        rdk_x = rdk_ecc + pos_jitter
    else:
        rdk_x = -rdk_ecc + pos_jitter
    rdk_y = pos_jitter
    rdk = visual.DotStim(
        win,
        nDots=25,
        dotSize=rdk_dotSize,
        speed=2,
        dotLife=40,
        dir=direction,
        coherence=coherence,
        fieldPos=(rdk_x, rdk_y),
        fieldSize=200,
        fieldShape='circle',
        signalDots='same',
        noiseDots="walk"
    )

    # send a message to log the onset of the stimulus
    el_tracker.sendMessage('rdk_onset')
    tar_onset_time = el_tracker.trackerTime()

    # define hit region (circle centered on RDK)
    # sac_tol_px = deg2pix(task_params["saccade_tolerance"], mon)
    sac_tol_px = 200
    hit_region = (int(scn_width/2.0 + rdk_x - sac_tol_px),
                  int(scn_height/2.0 - rdk_y - sac_tol_px),
                  int(scn_width/2.0 + rdk_x + sac_tol_px),
                  int(scn_height/2.0 - rdk_y + sac_tol_px))
    target_ia_msg = '!V IAREA RECTANGLE 1 %d %d %d %d rdk_IA' % hit_region
    el_tracker.sendMessage(target_ia_msg)
    el_tracker.sendCommand(f"draw_box {hit_region[0]} {hit_region[1]} {hit_region[2]} {hit_region[3]} 15")
    # fix_tol_px = 75
    fix_region = (int(scn_width/2.0 - fix_tol_px),
                  int(scn_height/2.0 - fix_tol_px),
                  int(scn_width/2.0 + fix_tol_px),
                  int(scn_height/2.0 + fix_tol_px))
    fix_ia_msg = '!V IAREA RECTANGLE 2 %d %d %d %d fix_IA' % fix_region
    el_tracker.sendMessage(fix_ia_msg)
    el_tracker.sendCommand(f"draw_box {fix_region[0]} {fix_region[1]} {fix_region[2]} {fix_region[3]} 15")
    # present the RDK for stim_duration while monitoring for saccades
    got_sac = False
    sac_start_time = -1
    SRT = -1
    acc = 0
    fix_dur = 0.1
    sac_timer = None
    sac_dur = task_params["saccade_duration"] / 1000.0

    event.clearEvents()
    start_time = time.time()
    while time.time() - start_time < rdk_dur and not got_sac:
        # draw fixation + RDK
        fix_cross.draw()
        rdk.draw()
        win.flip()
        if time.time() - start_time < fix_dur:
            continue

        # abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return 0

        # check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # immediate quit
            if keycode == 'escape':
                el_tracker.sendMessage('trial_skipped_by_user')
                clear_screen(win)
                abort_trial()
                return 0
            # Ctrl-C to terminate experiment (existing behavior)
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task()
                return 0
            # Pause the experiment mid-trial and return control to the outer loop
            if keycode == 'p':
                el_tracker.sendMessage('paused_by_user')
                # stop recording cleanly
                if el_tracker.isRecording():
                    pylink.pumpDelay(100)
                    el_tracker.stopRecording()
                clear_screen(win)
                return None
            # Calibrate immediately mid-trial (non-Ctrl 'c')
            if keycode == 'c' and not modifier.get('ctrl'):
                el_tracker.sendMessage('calibration_requested')
                # stop recording cleanly before calibration
                if el_tracker.isRecording():
                    pylink.pumpDelay(100)
                    el_tracker.stopRecording()
                try:
                    el_tracker.doTrackerSetup()
                except RuntimeError as err:
                    print('Calibration error during trial:', err)
                    try:
                        el_tracker.exitCalibration()
                    except Exception:
                        pass
                # return to outer loop so the experimenter can choose next action
                clear_screen(win)
                return None
            # give reward
            if keycode == 'r':
                give_reward()
                print("Reward given")

        # check for saccade events
        acc = 0
        # sac_tol_px = 100
        cx = scn_width/2.0 + rdk_x
        cy = scn_height/2.0 - rdk_y
        sac_hold = 10
        try:
            sample = el_tracker.getNewestSample()
        except Exception:
            sample = None

        gx = gy = None
        gaze_center_x = gaze_center_y = None

        if sample is not None:
            try:
                gaze = sample.getLeftEye().getGaze()
                if gaze is not None:
                    gx, gy = gaze
            except Exception:
                try:
                    gx = sample.gx
                    gy = sample.gy
                except Exception:
                    gx = gy = None

        if (gx is not None) and (gy is not None):
            gaze_center_x = gx - cx
            gaze_center_y = gy - cy
        else:
            gaze_center_x = gaze_center_y = None

        if (gaze_center_x is not None) and (gaze_center_y is not None):

            if cond == "go":
                # Check outside fixation
                dfx = gx - scn_width / 2.0
                dfy = gy - scn_height / 2.0
                if (dfx * dfx + dfy * dfy) >= (fix_tol_px * fix_tol_px):
                    if sac_timer is None:
                        sac_timer = time.time()
                    if time.time() - sac_timer >= sac_dur:
                        print(f"Eyes outside the window at {time.time() - sac_timer}")
                        # Check inside target
                        dx = gaze_center_x
                        dy = gaze_center_y
                        if (dx * dx + dy * dy) <= (sac_tol_px * sac_tol_px):
                            acc = 1
                            got_sac = True
                    else:
                        acc = 0
                        got_sac = False
            else:
                dx = gx - scn_width/2.0
                dy = gy - scn_height/2.0
                if (dx * dx + dy * dy) >= (fix_tol_px * fix_tol_px):
                    acc = 0
                    got_sac = True
                # if hold_start is None:
                #     hold_start = time.time()
                # elif (time.time() - hold_start) >= sac_hold:
                #     acc = 1
                #     got_sac = True
                #     break

        eye_ev = el_tracker.getNextData()
        if eye_ev == pylink.ENDSACC:
            print(eye_ev)
        # if (eye_ev is not None) and (eye_ev == pylink.ENDSACC):
        #     # if eye_ev is not None:
        #     eye_dat = el_tracker.getFloatData()
        #     if eye_dat.getEye() == eye_used:
        #         sac_amp = eye_dat.getAmplitude()
        #         sac_start_time = eye_dat.getStartTime()
        #         sac_end_pos = eye_dat.getEndGaze()

        #         # ignore saccades that start before stimulus onset
        #         if sac_start_time <= tar_onset_time:
        #             sac_start_time = -1
        #         elif hypot(sac_amp[0], sac_amp[1]) > 1.5:
        #             offset = int(el_tracker.trackerTime() - sac_start_time)
        #             el_tracker.sendMessage(f'{offset} saccade_resp')
        #             SRT = sac_start_time - tar_onset_time

        #             # For 'go' trials, correct if land in RDK hit region
        #             sac_x_pix = sac_end_pos[0]
        #             sac_y_pix = sac_end_pos[1]
        #             cx = scn_width/2.0 + rdk_x
        #             cy = scn_height/2.0 - rdk_y
        #             if cond == 'go':
        #                 if fabs(sac_x_pix - cx) < 100 and fabs(sac_y_pix - cy) < 100:
        #                     acc = 1
        #                 else:
        #                     acc = 0
        #             else:
        #                 # any saccade during no-go/noise is incorrect
        #                 acc = 0

        #             got_sac = True

    # For no-go and noise conditions, if no saccade happened during stim -> correct
    if (cond in ['no-go', 'noise']) and (got_sac is False):
        acc = 1

    # following the stimulus, hold for a short interval
    # core.wait(0.3)

    # following the saccadic response, show the target for an additional 300 ms
    # core.wait(0.3)

    # clear the screen
    clear_screen(win)
    el_tracker.sendMessage('blank_screen')
    # send a message to clear the Data Viewer screen as well
    el_tracker.sendMessage('!V CLEAR 128 128 128')

    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    el_tracker.stopRecording()

    # record trial variables to the EDF data file, for details, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('!V TRIAL_VAR condition %s' % cond)
    pylink.msecDelay(4)  # take a break of 4 millisecond
    el_tracker.sendMessage('!V TRIAL_VAR tar_onset_time %d' % tar_onset_time)
    el_tracker.sendMessage('!V TRIAL_VAR sac_start_time %d' % sac_start_time)
    el_tracker.sendMessage('!V TRIAL_VAR SRT %d' % SRT)
    el_tracker.sendMessage('!V TRIAL_VAR acc %d' % acc)

    # send a 'TRIAL_RESULT' message to mark the end of trial
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)

    return acc


# Step 5: Run trials continuously, allowing pause and on-the-fly selection
available_blocks = ['go', 'no-go', 'noise']
trial_index = 1
total_trials = 1000


def pause_menu(el_tracker):
    """Show a pause menu to allow experimenter to select upcoming trial type
    or perform calibration. Returns the new_mode (one of available_blocks)
    or None if the experiment should terminate. This function blocks until
    a valid choice is made.
    """
    menu_text = (
        "PAUSED - select upcoming trial type or action:\n\n"
        "g: GO\n"
        "n: NO-GO\n"
        "s: NOISE\n"
        "c: Calibrate tracker now\n"
        "q: Quit experiment\n\n"
        "Press the corresponding key to continue."
    )

    show_msg(win, menu_text, wait_for_keypress=False)

    valid_keys = {'g': 'go', 'n': 'no-go', 's': 'noise', 'q': 'quit', 'c': 'calib'}

    while True:
        keys = event.waitKeys(keyList=list(valid_keys.keys()) + ['escape'])
        if keys is None:
            continue
        k = keys[0]
        if k == 'escape' or k == 'q':
            return None
        choice = valid_keys.get(k)
        if choice == 'calib':
            try:
                # doTrackerSetup requires tracker not recording
                if el_tracker.isRecording():
                    pylink.pumpDelay(100)
                    el_tracker.stopRecording()
                el_tracker.doTrackerSetup()
            except RuntimeError as err:
                print('Calibration error:', err)
                try:
                    el_tracker.exitCalibration()
                except Exception:
                    pass
            # after calibration, show the menu again
            show_msg(win, menu_text, wait_for_keypress=False)
            continue
        elif choice in available_blocks:
            return choice


# initial selection for the running mode
intro_text = (
    "Select initial trial type:\n\n"
    "g: GO (saccade to upward-moving dots)\n"
    "n: NO-GO (maintain fixation for downward-moving dots)\n"
    "s: NOISE (maintain fixation for noise dots)\n\n"
    "During the experiment press 'p' to pause and choose upcoming trial types,\n"
    "press 'c' to calibrate at any time, or 'escape' to quit."
)
show_msg(win, intro_text, wait_for_keypress=False)
valid_keys = {'g': 'go', 'n': 'no-go', 's': 'noise'}
current_mode = None
while current_mode is None:
    keys = event.waitKeys(keyList=list(valid_keys.keys()) + ['escape'])
    if keys is None:
        continue
    k = keys[0]
    if k == 'escape':
        terminate_task()
    if k in valid_keys:
        current_mode = valid_keys[k]

if dummy_mode:
    print('ERROR: This task requires real-time gaze data.\nIt cannot run in Dummy mode.')
    terminate_task()

# run initial calibration before starting (same behavior as before)
try:
    el_tracker.doTrackerSetup()
except RuntimeError as err:
    print('ERROR:', err)
    try:
        el_tracker.exitCalibration()
    except Exception:
        pass

stim_params = read_config(config_dir / 'stimuli.yaml')
task_params = read_config(config_dir / 'task.yaml')
trial_history = []

# run trials continuously until we reach total_trials or the experimenter quits
while trial_index <= total_trials:
    # allow immediate termination between trials
    # run the trial with the current_mode; run_trial may return None to indicate
    # it was aborted (e.g., due to pause), 0/1 for accuracy, or pylink.TRIAL_ERROR
    acc = run_trial(current_mode, trial_index, stim_params, task_params)

    # if the trial was aborted (None), show pause menu to pick next mode
    if acc is None:
        choice = pause_menu(el_tracker)
        if choice is None:
            terminate_task()
        current_mode = choice
        # do not increment trial_index, re-run the same trial index with the new mode
        continue

    # reward on correct trials
    if acc == 1:
        give_reward()
        el_tracker.sendMessage('reward_given')
    else:
        el_tracker.sendMessage('no_reward')

    trial_history.append({
        'trial_index': trial_index,
        'mode': current_mode,
        'correct': acc
    })

    print(f"Mode: {current_mode}, correct: {acc}, accuracy: {sum(1 for t in trial_history if t['correct'] == 1)}/{len(trial_history)}")
    trial_index += 1
    core.wait(0.5)

# Step 6: disconnect, download the EDF file, then terminate the task
terminate_task()

# Save trial history to a file
df = pd.DataFrame(trial_history)
df.to_csv(data_dir / 'trial_history.csv', index=False)
