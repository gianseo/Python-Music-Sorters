#!/usr/bin/env python3
"""
Sort an iTunes Music Library XML playlist by a chosen Spotify track metadata attribute or local audio feature using Spotipy and Librosa.
Reads an iTunes Music Library XML (.xml), targets a playlist (optional), lets the user choose
an attribute (e.g., popularity, duration_ms, explicit, track_number, disc_number, release_date,
energy_local, brightness, percussiveness_zcr, percussiveness_onset, contrast, style_and_key_similarity,
bpm, music_genre, harmonic_content_key, timbral_changes, dynamic_changes) and sort direction.
Looks up each track's data on Spotify or analyzes the local file via Librosa, sorts the playlist accordingly,
and writes a new XML file with the playlist reordered. The new playlist name in the XML will
append ": sorted by <attribute>" to the original name.
Defaults to the only playlist if none is specified; auto-generates the output filename.
Verbose output shows fetched values and final sort order.
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import datetime
import urllib.parse

# Local audio analysis dependencies
import librosa
import numpy as np

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOauthError
from spotipy.exceptions import SpotifyException

# --- Embed your Spotify API credentials here ---
CLIENT_ID = 'XXXXXXXX'
CLIENT_SECRET = 'XXXXXXXX'

# Available attributes
ATTRIBUTES = [
    # Spotify metadata
    'popularity',     # 0-100 popularity score
    'release_date',   # album release date (YYYY-MM-DD or YYYY)
    # Local Librosa features
    'energy_local',           # RMS energy
    'brightness',             # spectral centroid
    'percussiveness_zcr',     # zero-crossing rate
    'percussiveness_onset',   # onset strength
    'contrast',               # spectral contrast
    'style_and_key_similarity', # tonal centroid (Tonnetz)
    'bpm',                    # estimated tempo
    'music_genre',            # MFCC average (proxy)
    'harmonic_content_key',   # chroma STFT average
    'timbral_changes',        # polynomial features average
    'dynamic_changes',        # MFCC deltas average
]

SORT_DIRECTIONS = {
    '1': ('ascending', False),
    '2': ('descending', True)
}

def load_library_tree(path):
    tree = ET.parse(path)
    root = tree.getroot()
    plist_dict = root.find('dict')
    tracks_map = {}
    for idx, elem in enumerate(list(plist_dict)):
        if elem.tag == 'key' and elem.text == 'Tracks':
            tracks_dict = list(plist_dict)[idx+1]
            entries = list(tracks_dict)
            i = 0
            while i < len(entries):
                if entries[i].tag == 'key':
                    tid = entries[i].text
                    info = entries[i+1]
                    data = {'Name': None, 'Artist': None, 'Location': None}
                    info_children = list(info)
                    for j in range(0, len(info_children), 2):
                        k = info_children[j].text
                        v = info_children[j+1].text
                        if k in data:
                            data[k] = v
                    tracks_map[tid] = data
                i += 2
            break
    return tree, plist_dict, tracks_map

def list_playlists(plist_dict):
    names = []
    for idx, elem in enumerate(list(plist_dict)):
        if elem.tag == 'key' and elem.text == 'Playlists':
            arr = list(plist_dict)[idx+1]
            for pl in list(arr):
                pts = list(pl)
                for k in range(len(pts)):
                    if pts[k].tag == 'key' and pts[k].text == 'Name':
                        names.append(pts[k+1].text)
            break
    return names

def find_playlist_dict(plist_dict, name):
    for idx, elem in enumerate(list(plist_dict)):
        if elem.tag == 'key' and elem.text == 'Playlists':
            arr = list(plist_dict)[idx+1]
            for pl in list(arr):
                pts = list(pl)
                for k in range(len(pts)):
                    if pts[k].tag == 'key' and pts[k].text == 'Name' and pts[k+1].text == name:
                        return pl
    return None

def get_playlist_track_ids(pl_dict):
    for idx, elem in enumerate(list(pl_dict)):
        if elem.tag == 'key' and elem.text == 'Playlist Items':
            arr = list(pl_dict)[idx+1]
            return [list(item)[1].text for item in list(arr) if list(item) and list(item)[0].text == 'Track ID']
    return []

def set_playlist_items(pl_dict, sorted_ids):
    for idx, elem in enumerate(list(pl_dict)):
        if elem.tag == 'key' and elem.text == 'Playlist Items':
            arr = list(pl_dict)[idx+1]
            for child in list(arr): arr.remove(child)
            for tid in sorted_ids:
                d = ET.SubElement(arr, 'dict')
                ET.SubElement(d, 'key').text = 'Track ID'
                ET.SubElement(d, 'integer').text = tid
            return

def fetch_value(sp, artist, title, location, attr):
    """
    Fetch a single attribute value: metadata via Spotipy, or local analysis via Librosa.
    """
    # Local Librosa features
    if location and attr.startswith(tuple(ATTRIBUTES[6:])):
        path = urllib.parse.unquote(location.replace('file://', ''))
        try:
            y, sr = librosa.load(path, sr=None)
        except Exception as e:
            print(f"[Local analysis error] {e}")
            return None
        # Compute each feature
        if attr == 'energy_local':
            return float(np.mean(librosa.feature.rms(y=y)))
        if attr == 'brightness':
            return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        if attr == 'percussiveness_zcr':
            return float(np.mean(librosa.feature.zero_crossing_rate(y)))
        if attr == 'percussiveness_onset':
            return float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
        if attr == 'contrast':
            return float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
        if attr == 'style_and_key_similarity':
            return float(np.mean(librosa.feature.tonnetz(y=y, sr=sr)))
        if attr == 'bpm':
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            return float(tempo)
        if attr == 'music_genre':
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            return float(np.mean(mfccs))
        if attr == 'harmonic_content_key':
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            return float(np.mean(chroma))
        if attr == 'timbral_changes':
            poly = librosa.feature.poly_features(y=y, sr=sr)
            return float(np.mean(poly))
        if attr == 'dynamic_changes':
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            delta = librosa.feature.delta(mfccs)
            return float(np.mean(delta))
    # Spotify metadata lookup
    q = []
    if artist: q.append(f'artist:"{artist}"')
    if title:  q.append(f'track:"{title}"')
    query = ' '.join(q) or title or artist
    try:
        res = sp.search(q=query, type='track', limit=1)
        items = res['tracks']['items']
        if not items:
            return None
        track = sp.track(items[0]['id'])
        if attr == 'release_date':
            rd = track.get('album', {}).get('release_date')
            if not rd: return None
            if len(rd) == 4:
                return datetime.date(int(rd),1,1)
            try:
                return datetime.date.fromisoformat(rd)
            except:
                return None
        return track.get(attr)
    except SpotifyException as e:
        print(f"[Spotify API Error] {e}")
        return None

def choose_attribute():
    print("Select a track attribute to sort by:")
    for i, a in enumerate(ATTRIBUTES, 1):
        print(f"  {i}. {a}")
    choice = input(f"Enter number or name (default 1 for '{ATTRIBUTES[0]}'): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(ATTRIBUTES):
        return ATTRIBUTES[int(choice)-1]
    if choice in ATTRIBUTES:
        return choice
    print(f"Invalid choice, defaulting to '{ATTRIBUTES[0]}'.")
    return ATTRIBUTES[0]

def choose_direction():
    print("Select sort direction:")
    print("  1. Ascending")
    print("  2. Descending")
    choice = input("Enter 1 or 2 (default 1): ").strip()
    return SORT_DIRECTIONS.get(choice, SORT_DIRECTIONS['1'])

def main():
    if not (CLIENT_ID and CLIENT_SECRET):
        print("✗ Please set CLIENT_ID and CLIENT_SECRET in the script.")
        sys.exit(1)
    parser = argparse.ArgumentParser(description='Sort an iTunes XML playlist by attribute or local audio feature.')
    parser.add_argument('input', help='Path to iTunes XML')
    parser.add_argument('playlist', nargs='?', help='Playlist to sort')
    parser.add_argument('output', nargs='?', help='Output XML path')
    args = parser.parse_args()

    tree, plist_dict, tracks_map = load_library_tree(args.input)
    pname = args.playlist or (list_playlists(plist_dict)[0] if len(list_playlists(plist_dict))==1 else None)
    if not pname:
        print(f"Available playlists: {list_playlists(plist_dict)}")
        pname = input("Enter playlist name to sort: ")
    pl_dict = find_playlist_dict(plist_dict, pname)
    if pl_dict is None:
        print(f"✗ Playlist '{pname}' not found.")
        sys.exit(1)
    tids = get_playlist_track_ids(pl_dict)
    if not tids:
        print(f"✗ No tracks in playlist '{pname}'.")
        sys.exit(1)

    attr = choose_attribute()
    dir_name, reverse = choose_direction()
    print(f"Sorting by: {attr} ({dir_name})\n")

    try:
        auth = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
        sp = spotipy.Spotify(auth_manager=auth)
    except SpotifyOauthError as e:
        print(f"✗ Spotify authentication failed: {e}")
        sys.exit(1)

    scored = []
    print("Fetching values:")
    for tid in tids:
        info = tracks_map.get(tid, {})
        val = fetch_value(sp, info.get('Artist'), info.get('Name'), info.get('Location'), attr)
        if val is None:
            print(f"⚠️ '{info.get('Name')}' missing '{attr}', placing last.")
            val = datetime.date.min if attr=='release_date' else float('inf')
        print(f"  {info.get('Name')}: {attr} = {val}")
        scored.append((tid, val))

    sorted_pairs = sorted(scored, key=lambda x: x[1], reverse=reverse)
    print("\nFinal order:")
    for i, (tid, v) in enumerate(sorted_pairs, 1):
        print(f"  {i}. {tracks_map[tid]['Name']} ({attr}={v})")

    sorted_ids = [tid for tid, _ in sorted_pairs]
    set_playlist_items(plist_dict, sorted_ids)

    # Update playlist name with suffix
    for idx, elem in enumerate(list(plist_dict)):
        if elem.tag == 'key' and elem.text == 'Playlists':
            arr = list(plist_dict)[idx+1]
            for pl in arr:
                name_elem = None
                pts = list(pl)
                for j in range(len(pts)):
                    if pts[j].tag == 'key' and pts[j].text == 'Name' and pts[j+1].text == pname:
                        name_elem = pts[j+1]
                        break
                if name_elem is not None:
                    name_elem.text = f"{pname} : sorted by {attr}"
                    break
            break

    out = args.output or os.path.splitext(args.input)[0] + '_sorted.xml'
    tree.write(out, encoding='utf-8', xml_declaration=True)
    print(f"\n✓ '{pname}' sorted by '{attr}' ({dir_name}) saved to {out}")

if __name__ == '__main__':
    main()
