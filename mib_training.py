#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =================================================================================================== #
#
#
#                     SCRIPT: mib_training.py
#
#
#                DESCRIPTION: Script to run training for the MIB task
#
#
#                       RULE: DAYW
#
#
#
#                    CREATOR: Sharif Saleki
#                       TIME: 09-19-2025-7810598105114117
#                      SPACE: Stanford Univeristy, Stanford, CA
#
# =================================================================================================== #
from pathlib import Path
import os
from datetime import date
import sys
import csv
import time

import numpy as np
import pandas as pd

import pylink
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, logging, gui
from psychopy.tools.monitorunittools import deg2pix

import nidaqmx
from nidaqmx.constants import VoltageUnits, AcquisitionType

# plotting handled by src.plotting.TrialPlotter

# Switch to the script folder
root = Path(__file__).parent.resolve()
os.chdir(root)
# Add the source directory to the path
src_dir = root / 'src'
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from src.experiment import clear_screen, show_msg  # noqa: E402 - after sys.path adjustment
from src.eyetracker import MyeLink  # noqa: E402 - after sys.path adjustment
from src.rdk import GaussianRDK  # noqa: E402 - after sys.path adjustment
from src.utils import load_yaml  # noqa: E402 - after sys.path adjustment
from src.plotting import TrialPlotter  # noqa: E402 - after sys.path adjustment


# Get info from the gui interface
today = date.today().strftime("%Y_%m_%d")
task_name = "RDK MIB"

dlg_title = 'Realtime RDK Training'
dlg_prompt = 'Enter session info ;)'
dlg = gui.Dlg(dlg_title)
dlg.addText(dlg_prompt)
dlg.addFixedField(key='Date', label="Date", initial=today)
dlg.addFixedField(key='Task', label="Task", initial=task_name)
dlg.addField(key="Subject", initial="AQ")
dlg.addField(key='Session', initial='')
dlg.addField(key="Debug", initial=False)

# show dialog and wait for OK or Cancel
ok_data = dlg.show()
if dlg.OK:  # if ok_data is not None
    sub_id = ok_data[2]
    ses_id = int(ok_data[3])
    debug = bool(ok_data[4])
else:
    print('User cancelled!')
    core.quit()

# Directories
assets_dir = root / 'assets'
config_dir = root / 'config'
data_dir = root / 'data' / today / f"sub-{sub_id}" / f"ses-{ses_id:02d}"
data_dir.mkdir(exist_ok=True, parents=True)
fig_dir = root / 'figures' / today / f"sub-{sub_id}" / f"ses-{ses_id:02d}"
fig_dir.mkdir(exist_ok=True, parents=True)
log_dir = root / 'logs' / today
log_dir.mkdir(exist_ok=True, parents=True)

# Files
data_file = str(data_dir / f"sub-{sub_id}_ses-{ses_id}_rdk_{today}.csv")
log_file = str(data_dir / f"sub-{sub_id}_ses-{ses_id}_rdk_{today}.log")
local_edf = str(data_dir / f"sub-{sub_id}_ses-{ses_id}_rdk_{today}.edf")
host_edf = f"rdk_ses_{ses_id}.edf"

calib_img_files = list(assets_dir.glob("*.png"))
target_beep = assets_dir / 'qbeep.wav'
good_beep = assets_dir / 'type.wav'
error_beep = assets_dir / 'error.wav'

# Load config
stim_params = load_yaml(config_dir / 'stimuli.yaml')
task_params = load_yaml(config_dir / 'task.yaml')
disp_params = load_yaml(config_dir / 'display.yaml')
mon_params = disp_params['Monitor']
win_params = disp_params['Window']
eyetracker_params = load_yaml(config_dir / 'eyetracker.yaml')
plotting_params = load_yaml(config_dir / 'plotting.yaml')

# Clocks
clock = core.Clock()

# Logging
logging.console.setLevel(logging.CRITICAL)
logging.LogFile(log_file, level=logging.INFO, filemode='w')
logging.setDefaultClock(clock)

# Define eyetracker interface
el_interface = MyeLink(config=eyetracker_params, debug=debug)
el_interface.connect()
el_interface.open_data_file(edf_filename=host_edf)
el_interface.add_preamble_text(f"RECORDED BY {task_name}")
el_interface.offline()
eyelink_version = el_interface.check_version()
el_interface.setup_data_filters()

# Monitor and window
screen_num = 1 if not debug else 0  # use the external display in non-debug mode
full_screen = True if not debug else False

all_monitors = monitors.getAllMonitors()
if mon_params['name'] not in all_monitors:
    mon = monitors.Monitor(
        mon_params['name'],
        width=mon_params['dimensions'][0],
        distance=mon_params['distance'],
        # gamma=None
    )
    mon.setSizePix(mon_params['resolution'])
    mon.save()
else:
    mon = monitors.Monitor(mon_params['name'])

win = visual.Window(
    # color=win_params["color"],
    fullscr=full_screen,
    screen=screen_num,
    monitor=mon,
    winType='pyglet',
    units=win_params['units'],
    waitBlanking=True,
    checkTiming=True
)

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
# see the EyeLink Installation Guide, "Customizing Screen Settings"
scn_width, scn_height = mon_params['resolution']
el_coords = f"screen_pixel_coords = 0 0 {scn_width - 1} {scn_height - 1}"
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization, see Data
# Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
dv_coords = f"DISPLAY_COORDS  0 0 {scn_width - 1} {scn_height - 1}"
el_tracker.sendMessage(dv_coords)

# Keep calibration/validation targets closer to the center
el_tracker.sendCommand(
    f"calibration_area_proportion {eyetracker_params['area_proportion'][0]} {eyetracker_params['area_proportion'][1]}"
)
el_tracker.sendCommand(
    f"validation_area_proportion {eyetracker_params['area_proportion'][0]} {eyetracker_params['area_proportion'][1]}"
)

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
logging.log(level=logging.INFO, msg=genv)  # print out the version number of the CoreGraphics library

# register a reward callback so each accepted calibration target can trigger reward
# try:
#     genv.set_reward_callback(lambda: give_reward())
# except Exception:
#     pass

# Set background and foreground colors for the calibration target
genv.setCalibrationColors(eyetracker_params['foreground_color'], eyetracker_params['background_color'])

# Set up the calibration target
genv.setTargetType('picture')
calib_images = list(np.random.choice(calib_img_files, int(eyetracker_params["calibration_type"][-1]), replace=False))
genv.setPictureTarget(calib_images)
genv.setTargetSize(int(deg2pix(eyetracker_params['target_size'], monitor=mon)))

# Beeps to play during calibration, validation and drift correction
genv.setCalibrationSounds(
    str(target_beep),
    str(good_beep),
    str(error_beep)
)

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)

# Initialize plotter for real-time updates
plotter = TrialPlotter(
    monitor=mon,
    subject=sub_id,
    session=ses_id,
    run_date=date.today().strftime('%Y-%m-%d'),
    config=plotting_params,
)

# (plotting moved into src/plotting.TrialPlotter)


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
    single = np.concatenate([
        np.ones(on_samples) * float(reward_voltage),
        np.zeros(off_samples)
    ])
    signal = np.tile(single, pulses)

    # fallback: ensure signal not empty
    if signal.size == 0:
        signal = np.array([float(reward_voltage)])

    try:
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


def terminate_task(history):
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
            el_tracker.receiveDataFile(host_edf, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # save data
    # Save trial history to a file
    df = pd.DataFrame(history)
    # df.to_csv(data_dir / 'trial_history.csv', index=False)
    try:
        # prefer pandas writer
        df.to_csv(data_dir / 'trial_history.csv', index=False)
    except Exception as e:
        # fallback: write CSV with stdlib to avoid pandas import internals
        out_path = data_dir / 'trial_history_fallback.csv'
        try:
            with open(out_path, 'w', newline='', encoding='utf-8') as fh:
                if len(df.columns) == 0:
                    fh.write('')  # empty frame
                else:
                    writer = csv.writer(fh)
                    writer.writerow(df.columns)
                    for row in df.itertuples(index=False, name=None):
                        writer.writerow(row)
            print(f"Warning: pandas.to_csv failed ({e}); wrote fallback CSV: {out_path}")
        except Exception as e2:
            print(f"Error: failed to save trial history with fallback method: {e2}")

    # save final plot if it exists
    try:
        if plotter is not None and plotter.figure() is not None:
            fig = plotter.figure()
            out_png = fig_dir / 'trial_summary.png'
            out_pdf = fig_dir / 'trial_summary.pdf'
            fig.savefig(str(out_png), bbox_inches='tight', dpi=150)
            fig.savefig(str(out_pdf), bbox_inches='tight', dpi=150)
            print(f"Saved plots: {out_png}, {out_pdf}")
    except Exception as e:
        print("Warning: failed to save plot:", e)

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
    mouse = event.Mouse(win=win, visible=False)

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


def run_trial(condition, trial_index, stim_params, task_params, history):
    """Run a single RDK trial.

    cond - one of 'go', 'no-go', or 'noise'
    trial_index - order index

    Returns acc (1=correct, 0=incorrect)
    """
    # RDK display parameters (position relative to center)
    rdk_params = stim_params["RDK"]
    noise_params = stim_params["Noise"]
    rdk_ecc = deg2pix(rdk_params["eccentricity"], mon)
    rdk_pos_jitter = deg2pix(rdk_params["position_jitter"], mon)
    rdk_side = 1 if rdk_params["side"] == "right" else 0
    jitter = np.random.uniform(-rdk_pos_jitter, rdk_pos_jitter)
    if rdk_side == 1:
        rdk_x = rdk_ecc + jitter
    else:
        rdk_x = -rdk_ecc + jitter
    rdk_y = jitter

    # Gaussian sigma defaults to half-radius; allow override via YAML (optional)
    rdk_sigma = deg2pix(rdk_params.get("gauss_sigma", rdk_params["field_size"]/2.0), mon)
    noise_sigma = deg2pix(noise_params.get("gauss_sigma", noise_params["field_size"]/2.0), mon)

    # Convert units to pixels
    dot_size_px = deg2pix(rdk_params["dot_size"], mon)
    speed_px = deg2pix(condition["speed"], mon)
    field_size_px = deg2pix(rdk_params["field_size"], mon)

    # Main RDK with coherent signal + per-frame random-direction noise
    rdk = GaussianRDK(
        win=win,
        n_dots=rdk_params["n_dots"],
        dot_size=dot_size_px,
        speed=speed_px,
        dot_life=rdk_params["dot_life"],
        direction=condition["direction"],
        coherence=condition["coherence"],
        field_pos=(rdk_x, rdk_y),
        field_size=field_size_px,
        gauss_sigma=rdk_sigma,
        color=(1, 1, 1),
    )

    # Background noise-only RDK (coherence=0), sharing same center and mask
    noise = GaussianRDK(
        win=win,
        n_dots=noise_params["n_dots"],
        dot_size=deg2pix(noise_params["dot_size"], mon),
        speed=speed_px,
        dot_life=noise_params["dot_life"],
        direction=np.random.uniform(0, 360),
        coherence=0.0,
        field_pos=(rdk_x, rdk_y),
        field_size=deg2pix(noise_params["field_size"], mon),
        gauss_sigma=noise_sigma,
        color=(1, 1, 1),
    )

    # Fixation stimulus
    fix_params = stim_params["Fixation"]
    fix_cross = visual.Circle(
        win,
        radius=deg2pix(fix_params["size"], mon),
        fillColor=fix_params["fill_color"],
        lineColor=fix_params["line_color"],
        lineWidth=fix_params["line_width"],
    )
    # fix_img = np.random.choice(calib_images)
    # fix_cross = visual.ImageStim(
    #     win,
    #     image=str(fix_img),
    #     size=deg2pix(fix_params["width"], mon),
    #     pos=(0, 0),
    #     units='pix',
    #     interpolate=True,
    #     color=(1, 1, 1),
    #     colorSpace='rgb',
    #     opacity=1.0,
    #     flipHoriz=False,
    #     flipVert=False,
    #     texRes=128,
    #     mask=None,
    # )

    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()

    # draw cross on the Host PC display to help visualization
    scn_width, scn_height = win.size
    cross_coords = (int(scn_width/2.0), int(scn_height/2.0))
    el_tracker.sendCommand('clear_screen 0')  # clear the host Display
    el_tracker.sendCommand('draw_cross %d %d 10' % cross_coords)

    # send a "TRIALID" message to mark the start of a trial
    el_tracker.sendMessage('TRIALID %d' % trial_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = f'TRIAL number {trial_index}'
    el_tracker.sendCommand(f"record_status_message '{status_msg}'")

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

    # Draw fixation cross guide in Data Viewer
    # h_center = int(scn_width/2.0)
    # v_center = int(scn_height/2.0)
    # line_hor = (h_center - 20, v_center, h_center + 20, v_center)
    # line_ver = (h_center, v_center - 20, h_center, v_center + 20)
    # el_tracker.sendMessage('!V CLEAR 128 128 128')  # clear the screen
    # el_tracker.sendMessage('!V DRAWLINE 0 255 0 %d %d %d %d' % line_hor)
    # el_tracker.sendMessage('!V DRAWLINE 0 255 0 %d %d %d %d' % line_ver)
    # Define hit region (circle centered on RDK)
    sac_tol_px = deg2pix(rdk_params["field_size"], mon) / 2.0 + deg2pix(task_params["saccade_tolerance"], mon)
    fix_tol_px = deg2pix(task_params["fixation_tolerance"], mon)
    hit_region = (
        int(scn_width/2.0 + rdk_x - sac_tol_px),
        int(scn_height/2.0 - rdk_y - sac_tol_px),
        int(scn_width/2.0 + rdk_x + sac_tol_px),
        int(scn_height/2.0 - rdk_y + sac_tol_px)
    )
    # target_ia_msg = '!V IAREA RECTANGLE 1 %d %d %d %d rdk_IA' % hit_region
    # el_tracker.sendMessage(target_ia_msg)
    el_tracker.sendCommand(f"draw_box {hit_region[0]} {hit_region[1]} {hit_region[2]} {hit_region[3]} 15")
    fix_region = (
        int(scn_width/2.0 - fix_tol_px),
        int(scn_height/2.0 - fix_tol_px),
        int(scn_width/2.0 + fix_tol_px),
        int(scn_height/2.0 + fix_tol_px)
    )
    # fix_ia_msg = '!V IAREA RECTANGLE 2 %d %d %d %d fix_IA' % fix_region
    # el_tracker.sendMessage(fix_ia_msg)
    el_tracker.sendCommand(f"draw_box {fix_region[0]} {fix_region[1]} {fix_region[2]} {fix_region[3]} 15")

    # Simple, non-blinking fixation acquire: check gaze samples for timeout
    fix_timeout = task_params["fixation_timeout"] / 1000.0
    fix_hold_time = task_params["fixation_hold"] / 1000.0
    blink_period = task_params["fixation_blink_period"] / 1000.0
    scn_width, scn_height = win.size
    cx = scn_width / 2.0
    cy = scn_height / 2.0
    rdk_cx = rdk_x + cx
    rdk_cy = -rdk_y + cy

    # Setup trial data record
    trial_data = {
        "trial_index": trial_index,
        "speed": condition["speed"],
        "coherence": condition["coherence"],
        "direction": condition["direction"],
        "side": rdk_params["side"],
        "saccade_rt": -1,
        "saccade_duration": -1,
        "correct": 0,
        "saccade_x": -1,
        "saccade_y": -1,
        "stim_x": rdk_x,
        "stim_y": rdk_y,
    }

    # Step 1: Acquire fixation
    if debug:
        # in dummy mode, just wait a moment and assume fixation acquired
        core.wait(0.5)
        fix_acquired = True

    start_t = clock.getTime()
    blink_start = clock.getTime()
    hold_start = None
    fix_acquired = False
    fix_on = True

    # Acquire fixation within timeout
    while (time.time() - start_t) < fix_timeout:

        # Abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return 0

        # Blink the fixation
        if time.time() - blink_start > blink_period:
            fix_on = not fix_on
            blink_start = time.time()

        # Draw fixation if on
        if fix_on:
            fix_cross.draw()
        win.flip()

        # Check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # immediate quit
            if keycode == 'escape':
                el_tracker.sendMessage('trial_skipped_by_user')
                clear_screen(win)
                abort_trial()
                return trial_data
            # Ctrl-C to terminate experiment (existing behavior)
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(history)
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
            # Give reward
            if keycode == 'r':
                give_reward()
                logging.info(f"Manual reward given on trial {trial_index}")

        # Try to get newest sample from EyeLink
        if debug:
            if time.time() - start_t > 0.5:
                # in simulation mode, just wait 0.5 seconds and assume fixation acquired
                fix_acquired = True
                break

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
            if (gaze_center_x * gaze_center_x + gaze_center_y * gaze_center_y) <= (fix_tol_px * fix_tol_px):
                if hold_start is None:
                    hold_start = time.time()
                elif (time.time() - hold_start) >= fix_hold_time:
                    fix_acquired = True
                    break
            else:
                hold_start = None

        # Small delay to prevent tight spin
        pylink.pumpDelay(10)

    # Terminate trial if fixation not acquired
    if not fix_acquired:
        # fixation failed: mark trial incorrect and skip
        el_tracker.sendMessage('fixation_not_acquired')
        clear_screen(win)
        if el_tracker.isRecording():
            pylink.pumpDelay(50)
            el_tracker.stopRecording()
        return trial_data

    # Step 2: Present fixation + noise RDK
    fix_dur = task_params["fixation_duration"] / 1000.0
    fix_jitter = task_params["fixation_jitter"] / 1000.0
    fix_dur = np.random.uniform(fix_dur - fix_jitter, fix_dur + fix_jitter)
    fixated = True

    fix_start_t = time.time()
    while time.time() - fix_start_t < fix_dur and fixated:
        
        # Abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return 0

        # Check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # immediate quit
            if keycode == 'escape':
                el_tracker.sendMessage('trial_skipped_by_user')
                clear_screen(win)
                abort_trial()
                return trial_data
            # Ctrl-C to terminate experiment (existing behavior)
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(history)
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
            # Give reward
            if keycode == 'r':
                give_reward()
                logging.info(f"Manual reward given on trial {trial_index}")

        # Draw fixation + Gaussian-masked noise field
        fix_cross.draw()
        noise.draw()
        win.flip()

        # Check for fixation break
        if debug:
            if time.time() - start_t > 0.5:
                # in simulation mode, just wait 0.5 seconds and assume fixation acquired
                fixated = True
                break

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
            if (gaze_center_x * gaze_center_x + gaze_center_y * gaze_center_y) > (fix_tol_px * fix_tol_px):
                fixated = False
                break

        # Small delay to prevent tight spin
        pylink.pumpDelay(10)

    if not fixated:
        # fixation break: mark trial incorrect and skip
        el_tracker.sendMessage('fixation_break_before_stimulus')
        clear_screen(win)
        if el_tracker.isRecording():
            pylink.pumpDelay(50)
            el_tracker.stopRecording()
        return trial_data

    # Step 3: Show RDK stimulus and check for saccade
    sac_init_dur = task_params["saccade_initiation_duration"] / 1000.0
    got_sac = False
    stim_onset = None
    
    start_t = time.time()
    
    # Loop until saccade_window elapsed
    while time.time() - start_t < sac_init_dur and not got_sac:

        # Abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return 0

        # Check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # immediate quit
            if keycode == 'escape':
                el_tracker.sendMessage('trial_skipped_by_user')
                clear_screen(win)
                abort_trial()
                return trial_data
            # Ctrl-C to terminate experiment (existing behavior)
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(history)
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
            # Give reward
            if keycode == 'r':
                give_reward()
                logging.info(f"Manual reward given on trial {trial_index}")
        
        # Draw Gaussian-masked RDK
        rdk.draw()
        win.flip()
        if stim_onset is None:
            stim_onset = time.time()

        # Check for saccade events
        if debug:
            if time.time() - start_t > 0.3:
                # in simulation mode, just wait 0.5 seconds and assume saccade made
                got_sac = True
                trial_data["saccade_rt"] = np.random.normal(250, 50)  # dummy value
                break

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
            if (gaze_center_x * gaze_center_x + gaze_center_y * gaze_center_y) > (fix_tol_px * fix_tol_px):
                got_sac = True
                trial_data["saccade_rt"] = (time.time() - stim_onset) * 1000.0  # in ms
                break

    # Abort if no saccade detected
    if not got_sac:
        el_tracker.sendMessage('no_saccade_detected')
        clear_screen(win)
        if el_tracker.isRecording():
            pylink.pumpDelay(50)
            el_tracker.stopRecording()
        return trial_data

    # Step 4: Check if saccade landed in target area
    sac_dur = task_params["saccade_duration"] / 1000.0
    landed_in_target = False
    start_t = time.time()

    while time.time() - start_t < sac_dur and not landed_in_target:

        # Abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return 0

        # Check for keyboard events
        for keycode, modifier in event.getKeys(modifiers=True):
            # immediate quit
            if keycode == 'escape':
                el_tracker.sendMessage('trial_skipped_by_user')
                clear_screen(win)
                abort_trial()
                return trial_data
            # Ctrl-C to terminate experiment (existing behavior)
            if keycode == 'c' and (modifier['ctrl'] is True):
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(history)
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
            # Give reward
            if keycode == 'r':
                give_reward()
                logging.info(f"Manual reward given on trial {trial_index}")

        # Draw Gaussian-masked RDK
        rdk.draw()
        win.flip()

        # Check for saccade events
        if debug:
            if time.time() - start_t > 0.1:
                landed_in_target = True
                trial_data["saccade_duration"] = np.random.normal(100, 30)  # dummy value
                trial_data["saccade_x"] = rdk_x + np.random.normal(-sac_tol_px/2, sac_tol_px/2)
                trial_data["saccade_y"] = rdk_y + np.random.normal(-sac_tol_px/2, sac_tol_px/2)
                trial_data["correct"] = 1
                break

        try:
            sample = el_tracker.getNewestSample()
            eye_ev = el_tracker.getNextData()
            if eye_ev != 200:
                if eye_ev == pylink.STARTBLINK:
                    print(f"Found event: {eye_ev} (blink start)")
                elif eye_ev == pylink.ENDBLINK:
                    print(f"Found event: {eye_ev} (blink end)")
                elif eye_ev == pylink.FIXUPDATE:
                    print(f"Found event: {eye_ev} (fixation update)")
                elif eye_ev == pylink.STARTSACCADE:
                    print(f"Found event: {eye_ev} (saccade start)")
                elif eye_ev == pylink.ENDSACCADE:
                    print(f"Found event: {eye_ev} (saccade end)")
                else:
                    print(f"Found event: {eye_ev} (other)")
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
            gaze_center_x = gx - rdk_cx
            gaze_center_y = -gy + rdk_cy
        else:
            gaze_center_x = gaze_center_y = None

        if (gaze_center_x is not None) and (gaze_center_y is not None):
            if (gaze_center_x * gaze_center_x + gaze_center_y * gaze_center_y) <= (sac_tol_px * sac_tol_px):
                landed_in_target = True
                trial_data["correct"] = 1
                trial_data["saccade_duration"] = time.time() - start_t
                trial_data["saccade_x"] = gx - cx
                trial_data["saccade_y"] = -gy + cy
                # print(f"gaze_center_x: {gaze_center_x}, gaze_center_y: {gaze_center_y}")
                # print(f"Target x: {rdk_x}, Target y: {rdk_y}, Target center: ({rdk_cx}, {rdk_cy})")
                break

    # Check if the saccade was correct
    if not landed_in_target:
        el_tracker.sendMessage('saccade_landed_outside_target')
        clear_screen(win)
        if el_tracker.isRecording():
            pylink.pumpDelay(50)
            el_tracker.stopRecording()
        return trial_data

    # clear the screen
    clear_screen(win)
    el_tracker.sendMessage('blank_screen')

    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    el_tracker.stopRecording()

    # send a 'TRIAL_RESULT' message to mark the end of trial
    el_tracker.sendMessage(f'TRIAL_RESULT {pylink.TRIAL_OK}')

    return trial_data


def pause_menu(el_tracker):
    """Show a pause menu to allow experimenter to select upcoming trial type
    or perform calibration. Returns the new_mode (one of available_blocks)
    or None if the experiment should terminate. This function blocks until
    a valid choice is made.
    """
    menu_text = (
        "PAUSED\n\n"
        "space: Resume\n"
        "c: Calibrate tracker now\n"
        "q: Quit experiment\n\n"
        "Press the corresponding key to continue."
    )

    show_msg(win, menu_text, wait_for_keypress=False)

    valid_keys = {'q': 'quit', 'c': 'calib', 'space': 'resume'}

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
        else:
            return choice


# ------------------------------
# Run trials continuously, allowing pause and on-the-fly selection
trial_index = 1
total_trials = 10000
trial_history = []

# Intro
intro_text = (
    "Press 'p' to pause.\n"
    "Press 'c' to calibrate at any time, or 'escape' to quit."
)
show_msg(win, intro_text, wait_for_keypress=False)

# if debug:
#     print('ERROR: This task requires real-time gaze data.\nIt cannot run in Dummy mode.')
#     terminate_task(trial_history)

# run initial calibration before starting (same behavior as before)
try:
    el_tracker.doTrackerSetup()
except RuntimeError as err:
    print('ERROR:', err)
    try:
        el_tracker.exitCalibration()
    except Exception:
        pass

# Make all trials with all combinations of conditions and repeat and shuffle them
all_trials = []

for speed in task_params["rdk_speeds"]:
    for coherence in task_params["coherence_levels"]:
        for direction in task_params["directions"]:
            all_trials.append({
                "speed": speed,
                "coherence": coherence,
                "direction": direction,
            })

all_trials = all_trials * task_params.get("n_trials_per_condition", 10)
np.random.shuffle(all_trials)

# Run
n_correct_in_a_row = 0
# while trial_index <= total_trials:
for trial_index, trial_conditions in enumerate(all_trials, start=1):

    # Run trial
    results = run_trial(trial_conditions, trial_index, stim_params, task_params, trial_history)

    # if the trial was aborted (None), show pause menu to pick next mode
    if results is None:
        choice = pause_menu(el_tracker)
        if choice is None:
            terminate_task(trial_history)
        continue

    # reward on correct trials
    if results["correct"] == 1:
        give_reward()
        el_tracker.sendMessage('reward_given')
        n_correct_in_a_row += 1
    else:
        el_tracker.sendMessage('no_reward')
        n_correct_in_a_row = 0

    # Change mode automatically if 5 correct in a row
    trial_history.append(results)

    # Plot summary
    n_trials = len(trial_history)
    n_correct = sum(1 for t in trial_history if t['correct'] == 1)
    percent_correct = 100 * n_correct / n_trials if n_trials > 0 else 0
    # find how many correct trials in a row
    n_correct_in_a_row_total = 0
    for t in reversed(trial_history):
        if t['correct'] == 1:
            n_correct_in_a_row_total += 1
        else:
            break

    # compact status message (split to avoid overly long line)
    # print(f"RT: {results['saccade_rt']}, rewarded: {results['correct']}, ")
    # print(
    #     f"Consecutive correct: {n_correct_in_a_row_total}, "
    #     f"Total correct: {n_correct}, Percent: {percent_correct:.2f}%"
    # )

    # Plot
    if plotter is not None:
        plotter.update(trial_history)

    # Next trial
    trial_index += 1
    core.wait(task_params["inter_trial_interval"] / 1000.0)

# Step 6: disconnect, download the EDF file, then terminate the task
terminate_task(trial_history)
