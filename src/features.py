"""
features.py

Feature extraction from raw KMT (Keyboard, Mouse, Touchscreen) event streams.
Converts raw JSON events into a fixed-length feature vector per session.
"""

import numpy as np


def extract_keystroke_features(key_events):
    """
    Extract dwell time and flight time features from raw key events.

    Dwell time  = duration between key pressed and key released (same key)
    Flight time = gap between key released and next key pressed (transition speed)

    Parameters
    ----------
    key_events : list of dict
        Raw key events from the JSON dataset.

    Returns
    -------
    dict of float features
    """
    # Filter clean events only
    key_events = [e for e in key_events if 'Event' in e and 'Epoch' in e]

    pressed = {}
    dwell_times = []
    release_epochs = []
    press_epochs = []
    backspace_count = 0

    for event in key_events:
        key = event['Key']
        epoch = float(event['Epoch'])
        action = event['Event']

        if action == 'pressed':
            press_epochs.append(epoch)
            if key in ('backspace', 'Backspace'):
                backspace_count += 1
            pressed[key] = epoch

        elif action == 'released' and key in pressed:
            dwell = epoch - pressed[key]
            if 0 <= dwell <= 2.0:
                dwell_times.append(dwell)
            release_epochs.append(epoch)
            del pressed[key]

    # Flight times from consecutive release -> press pairs
    release_epochs_sorted = sorted(release_epochs)
    press_epochs_sorted = sorted(press_epochs)

    flight_times = []
    for i in range(min(len(release_epochs_sorted), len(press_epochs_sorted)) - 1):
        flight = press_epochs_sorted[i+1] - release_epochs_sorted[i]
        if 0 < flight < 2.0:
            flight_times.append(flight)

    all_epochs = [float(e['Epoch']) for e in key_events]
    duration = max(all_epochs) - min(all_epochs) if len(all_epochs) > 1 else 0

    return {
        'dwell_mean':     np.mean(dwell_times)   if dwell_times   else 0,
        'dwell_std':      np.std(dwell_times)    if dwell_times   else 0,
        'dwell_min':      np.min(dwell_times)    if dwell_times   else 0,
        'dwell_max':      np.max(dwell_times)    if dwell_times   else 0,
        'dwell_median':   np.median(dwell_times) if dwell_times   else 0,
        'flight_mean':    np.mean(flight_times)  if flight_times  else 0,
        'flight_std':     np.std(flight_times)   if flight_times  else 0,
        'flight_min':     np.min(flight_times)   if flight_times  else 0,
        'flight_max':     np.max(flight_times)   if flight_times  else 0,
        'flight_median':  np.median(flight_times)if flight_times  else 0,
        'key_event_count': len(key_events),
        'backspace_rate': backspace_count / len(key_events) if key_events else 0,
        'key_duration':   duration
    }


def extract_mouse_features(mouse_events):
    """
    Extract velocity, trajectory, click dwell, and hover features
    from raw mouse events.

    Parameters
    ----------
    mouse_events : list of dict
        Raw mouse events from the JSON dataset.

    Returns
    -------
    dict of float features
    """
    # Filter clean events — drop malformed metadata records
    mouse_events = [e for e in mouse_events if 'Event' in e and 'Epoch' in e]

    movement_events = [e for e in mouse_events if e['Event'] == 'movement']
    click_press_events = [e for e in mouse_events if 'press' in e['Event']]

    # Velocity and trajectory
    velocities = []
    trajectory_length = 0.0

    for i in range(1, len(movement_events)):
        prev = movement_events[i-1]
        curr = movement_events[i]
        dx = curr['Coordinates'][0] - prev['Coordinates'][0]
        dy = curr['Coordinates'][1] - prev['Coordinates'][1]
        dt = float(curr['Epoch']) - float(prev['Epoch'])
        distance = np.sqrt(dx**2 + dy**2)
        trajectory_length += distance
        if dt > 0:
            velocities.append(distance / dt)

    # Directness ratio
    directness = 0.0
    if len(movement_events) >= 2 and trajectory_length > 0:
        first = movement_events[0]['Coordinates']
        last = movement_events[-1]['Coordinates']
        straight_line = np.sqrt(
            (last[0] - first[0])**2 + (last[1] - first[1])**2
        )
        directness = straight_line / trajectory_length

    # Click dwell
    click_dwells = []
    press_dict = {}
    for e in mouse_events:
        event_type = e['Event']
        epoch = float(e['Epoch'])
        if 'press' in event_type:
            press_dict[event_type.split()[0]] = epoch
        elif 'release' in event_type:
            button = event_type.split()[0]
            if button in press_dict:
                dwell = epoch - press_dict[button]
                if 0 <= dwell <= 2.0:
                    click_dwells.append(dwell)
                del press_dict[button]

    all_epochs = [float(e['Epoch']) for e in mouse_events]
    duration = max(all_epochs) - min(all_epochs) if len(all_epochs) > 1 else 0

    return {
        'mouse_velocity_mean':   np.mean(velocities)   if velocities    else 0,
        'mouse_velocity_std':    np.std(velocities)    if velocities    else 0,
        'mouse_velocity_max':    np.max(velocities)    if velocities    else 0,
        'mouse_velocity_median': np.median(velocities) if velocities    else 0,
        'trajectory_length':     trajectory_length,
        'directness_ratio':      directness,
        'click_dwell_mean':      np.mean(click_dwells) if click_dwells  else 0,
        'click_dwell_std':       np.std(click_dwells)  if click_dwells  else 0,
        'click_count':           len(click_press_events),
        'mouse_event_count':     len(mouse_events),
        'mouse_duration':        duration
    }


def extract_session_features(session, user_id, session_id, label):
    """
    Combine keystroke, mouse, and session-level features into a
    single fixed-length feature vector.

    Parameters
    ----------
    session : dict
        Raw session dict with 'key_events' and 'mouse_events' keys.
    user_id : str
        User identifier.
    session_id : str
        Session identifier (e.g. 'test_1').
    label : str
        'legitimate' or 'impostor'.

    Returns
    -------
    dict — one row ready for a pandas DataFrame
    """
    # Extract false_enters before filtering mouse events
    false_enters = 0
    for e in session['mouse_events']:
        if 'false_enters' in e:
            false_enters = e['false_enters']

    keystroke_features = extract_keystroke_features(session['key_events'])
    mouse_features = extract_mouse_features(session['mouse_events'])

    return {
        'user_id':    user_id,
        'session_id': session_id,
        'label':      label,
        **keystroke_features,
        **mouse_features,
        'false_enters': false_enters
    }