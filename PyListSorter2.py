#!/usr/bin/env python3
"""
Sort an iTunes Music Library XML playlist by a chosen embedded metadata attribute,
Spotify popularity, or local audio features using Mutagen (ID3 & MP4), Spotipy, and Librosa.
"""
import sys
import subprocess
import os
import argparse
import xml.etree.ElementTree as ET
import urllib.parse
import warnings

# suppress librosa/audioread and numpy warnings
warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import librosa
from mutagen import File as MutagenFile
import mutagen.id3 as id3
from mutagen.mp4 import MP4
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# --- Spotify credentials ---
CLIENT_ID = 'INSERTYOURSHERE'
CLIENT_SECRET = 'INSERTYOURSHERE'

# Grouped attributes
GROUPED_ATTRIBUTES = [
    ("Embedded Metadata", [
        ('1', 'Track BPM', 'beats_per_minute'),
        ('2', 'Danceability', 'mood_danceability'),
        ('3', 'Musical Key', 'initial_key'),
        ('4', 'MOOD AROUSAL', 'mood_arousal'),
        ('5', 'Color', 'beatunes_color'),
    ]),
    ("Local Analysis", [
        ('6', 'RMS energy', 'energy_local'),
        ('7', 'Spectral centroid', 'brightness'),
        ('8', 'Zero-crossing rate', 'percussiveness_zcr'),
        ('9', 'Onset strength', 'percussiveness_onset'),
        ('10', 'Spectral contrast', 'contrast'),
        ('11', 'Tonnetz (style/key)', 'style_and_key_similarity'),
        ('12', 'Beat tracker tempo', 'bpm'),
        ('13', 'Music genre proxy', 'music_genre'),
        ('14', 'Harmonic content key', 'harmonic_content_key'),
        ('15', 'Timbral changes', 'timbral_changes'),
        ('16', 'Dynamic changes', 'dynamic_changes'),
    ]),
    ("Spotify", [
        ('17', 'Spotify popularity', 'popularity'),
    ])
]
# Map attribute key to human label
ATTR_LABELS = { key: label for _, items in GROUPED_ATTRIBUTES for (_, label, key) in items }

# Sort directions
SORT_DIRECTIONS = {'1': ('ascending', False), '2': ('descending', True)}

# Map embedded attribute to tags
EMBEDDED_TAG_MAP = {
    'beats_per_minute': ['TBPM'],
    'mood_danceability': ['MOOD_DANCEABILITY', 'DanceabilityAlgorithm'],
    'initial_key': ['TKEY', 'initialkey'],
    'mood_arousal': ['MOOD_AROUSAL'],
    'beatunes_color': ['beaTunes_COLOR'],
}

# -----------------------------------------------------------------------
# Dependency check
# -----------------------------------------------------------------------
def check_dependencies():
    import importlib.util
    required = ['mutagen', 'librosa', 'numpy', 'spotipy']
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        resp = input(f"Missing: {', '.join(missing)}. Install? [y/N]: ").strip().lower()
        if resp == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("Installed. Restart script.")
            sys.exit(0)
        else:
            print("Exiting.")
            sys.exit(1)

check_dependencies()

# -----------------------------------------------------------------------
# XML Utilities
# -----------------------------------------------------------------------
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for c in elem:
            indent(c, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# -----------------------------------------------------------------------
# iTunes XML Parsing
# -----------------------------------------------------------------------
def load_library_tree(path):
    tree = ET.parse(path)
    root = tree.getroot()
    plist = root.find('dict')
    tracks = {}
    for idx, elem in enumerate(list(plist)):
        if elem.tag == 'key' and elem.text == 'Tracks':
            entries = list(list(plist)[idx+1])
            for key_elem, val_elem in zip(entries[::2], entries[1::2]):
                tid = key_elem.text
                info = {'Name': None, 'Artist': None, 'Location': None}
                children = list(val_elem)
                for a, b in zip(children[::2], children[1::2]):
                    if a.text in info:
                        info[a.text] = b.text
                tracks[tid] = info
            break
    return tree, plist, tracks

# -----------------------------------------------------------------------
# Playlist Helpers
# -----------------------------------------------------------------------
def list_playlists(plist):
    names = []
    for idx, elem in enumerate(list(plist)):
        if elem.tag == 'key' and elem.text == 'Playlists':
            for pl in list(plist)[idx+1]:
                pts = list(pl)
                for a, b in zip(pts[::2], pts[1::2]):
                    if a.text == 'Name':
                        names.append(b.text)
            break
    return names

def find_playlist_dict(plist, name):
    for idx, elem in enumerate(list(plist)):
        if elem.tag == 'key' and elem.text == 'Playlists':
            for pl in list(plist)[idx+1]:
                pts = list(pl)
                for a, b in zip(pts[::2], pts[1::2]):
                    if a.text == 'Name' and b.text == name:
                        return pl
    return None

def get_playlist_track_ids(pl_dict):
    for idx, elem in enumerate(list(pl_dict)):
        if elem.tag == 'key' and elem.text == 'Playlist Items':
            return [item.find('integer').text for item in list(pl_dict)[idx+1]]
    return []

def set_playlist_items(pl_dict, sorted_ids):
    for idx, elem in enumerate(list(pl_dict)):
        if elem.tag == 'key' and elem.text == 'Playlist Items':
            arr = list(pl_dict)[idx+1]
            for child in list(arr):
                arr.remove(child)
            for tid in sorted_ids:
                d = ET.SubElement(arr, 'dict')
                ET.SubElement(d, 'key').text = 'Track ID'
                ET.SubElement(d, 'integer').text = tid
            return

# -----------------------------------------------------------------------
# Embedded Metadata Fetching
# -----------------------------------------------------------------------
def fetch_embedded(path, descriptors, attr=None):
    ext = os.path.splitext(path)[1].lower()
    # MP4/M4A
    if ext in ('.m4a', '.mp4', '.m4b'):
        try:
            mp4f = MP4(path)
            tags = mp4f.tags or {}
        except Exception as e:
            print(f"[MP4 load error] {e}")
            return None
        # BPM
        if attr == 'beats_per_minute':
            bpm = tags.get('tmpo')
            if bpm:
                return float(bpm[0])
        # custom tags
        for desc in descriptors:
            key = f'----:com.apple.iTunes:{desc}'
            if key in tags:
                raw = tags[key][0]
                if attr == 'mood_arousal':
                    try:
                        raw_str = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
                        parts = raw_str.split(';')
                        return float(parts[1])
                    except Exception:
                        try:
                            return float(parts[0])
                        except Exception:
                            return raw_str
                return _convert_raw(desc, raw)
        return None
    # ID3 (MP3)
    try:
        id3tags = id3.ID3(path)
    except Exception:
        return None
    for desc in descriptors:
        for frame in id3tags.getall(desc):
            raw = getattr(frame, 'text', [None])[0] or getattr(frame, 'value', None)
            if raw is not None:
                return _convert_raw(desc, raw)
        for frame in id3tags.getall('TXXX'):
            if frame.desc == desc and frame.text:
                return _convert_raw(desc, frame.text[0])
    return None

def _convert_raw(tag, raw):
    try:
        return float(raw)
    except Exception:
        return str(raw)

# -----------------------------------------------------------------------
# Unified Value Fetch
# -----------------------------------------------------------------------
def fetch_value(sp, artist, title, location, attr):
    path = urllib.parse.unquote(location.replace('file://', '')) if location else None
    if path and attr in EMBEDDED_TAG_MAP:
        val = fetch_embedded(path, EMBEDDED_TAG_MAP[attr], attr)
        if val is not None:
            return val
    if attr == 'popularity':
        q = []
        if artist: q.append(f'artist:"{artist}"')
        if title:  q.append(f'track:"{title}"')
        query = ' '.join(q) or title or artist
        try:
            res = sp.search(q=query, type='track', limit=1)
            items = res['tracks']['items']
            if items:
                return sp.track(items[0]['id']).get('popularity')
        except SpotifyException as e:
            print(f"[Spotify error] {e}")
    if path and attr != 'popularity':
        try:
            y, sr = librosa.load(path, sr=None)
        except Exception as e:
            print(f"[Librosa load error] {e}")
            return None
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
            return float(np.mean(librosa.feature.mfcc(y=y, sr=sr)))
        if attr == 'harmonic_content_key':
            return float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
        if attr == 'timbral_changes':
            return float(np.mean(librosa.feature.poly_features(y=y, sr=sr)))
        if attr == 'dynamic_changes':
            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            return float(np.mean(librosa.feature.delta(mfccs)))
    return None

# -----------------------------------------------------------------------
# User Interaction & Main
# -----------------------------------------------------------------------
def choose_attribute():
    print("Select an attribute to sort by:")
    for group, items in GROUPED_ATTRIBUTES:
        print(f"\n{group}:")
        for num, label, _ in items:
            print(f"  {num}. {label}")
    choice = input("\nEnter choice number (default 1): ").strip()
    for _, items in GROUPED_ATTRIBUTES:
        for num, _, attr in items:
            if choice == num:
                return attr
    print("Invalid choice, defaulting to 'beats_per_minute'")
    return 'beats_per_minute'


def choose_direction():
    print("Select sort direction:")
    print("  1. Ascending")
    print("  2. Descending")
    choice = input("Enter 1 or 2 (default 1): ").strip()
    return SORT_DIRECTIONS.get(choice, SORT_DIRECTIONS['1'])


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("✗ Set CLIENT_ID and CLIENT_SECRET in the script.")
        sys.exit(1)
    parser = argparse.ArgumentParser(description='Sort an iTunes XML playlist by attribute.')
    parser.add_argument('input', help='Path to iTunes Library XML')
    parser.add_argument('playlist', nargs='?', help='Playlist name')
    parser.add_argument('output', nargs='?', help='Output XML path')
    args = parser.parse_args()

    tree, plist, tracks_map = load_library_tree(args.input)
    playlists = list_playlists(plist)
    pname = args.playlist or (playlists[0] if len(playlists) == 1 else None)
    if not pname:
        print(f"Available playlists: {playlists}")
        pname = input("Enter playlist name: ")
    pl = find_playlist_dict(plist, pname)
    if not pl:
        print(f"✗ Playlist '{pname}' not found.")
        sys.exit(1)

    tids = get_playlist_track_ids(pl)
    if not tids:
        print(f"✗ No tracks in '{pname}'.")
        sys.exit(1)

    attr = choose_attribute()
    dir_name, rev = choose_direction()
    print(f"\nSorting by {ATTR_LABELS.get(attr, attr)} ({dir_name})\n")

    auth = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth)

    # Fetch or calculate values
    scored = []
    local_keys = [key for group, items in GROUPED_ATTRIBUTES if group == "Local Analysis" for key in [k for (_,_,k) in items]]
    if attr in local_keys:
        print("Calculating values:")
    else:
        print("Fetching values:")
    for tid in tids:
        info = tracks_map.get(tid, {})
        val = fetch_value(sp, info.get('Artist'), info.get('Name'), info.get('Location'), attr)
        if val is None:
            print(f"⚠️ '{info.get('Name')}' missing '{attr}', placing last.")
            val = float('inf')
        print(f"  {info.get('Name')}: {val}")
        scored.append((tid, val))

    # Sort
    def _sort_key(item):
        v = item[1]
        return (0, v) if isinstance(v, (int, float)) else (1, str(v).lower())
    sorted_pairs = sorted(scored, key=_sort_key, reverse=rev)
    sorted_ids = [tid for tid, _ in sorted_pairs]
    set_playlist_items(pl, sorted_ids)

    # Rename playlist
    for idx, elem in enumerate(list(plist)):
        if elem.tag == 'key' and elem.text == 'Playlists':
            for p in list(plist)[idx+1]:
                pts = list(p)
                for j in range(len(pts)):
                    if pts[j].tag == 'key' and pts[j].text == 'Name' and pts[j+1].text == pname:
                        label = ATTR_LABELS.get(attr, attr)
                        pts[j+1].text = f"{pname} : sorted by {label}"
    out = args.output or os.path.splitext(args.input)[0] + '_sorted.xml'
    if os.path.exists(out):
        if input(f"\nOverwrite '{out}'? [y/N]: ").strip().lower() != 'y':
            sys.exit(0)
    indent(tree.getroot())
    tree.write(out, encoding='utf-8', xml_declaration=True)
    print(f"\n✓ Saved '{pname}' sorted by {ATTR_LABELS.get(attr, attr)} to {out}")

if __name__ == '__main__':
    main()
