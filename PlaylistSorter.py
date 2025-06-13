#!/usr/bin/env python3
"""
Sort an iTunes Music Library XML playlist by a chosen Spotify track metadata attribute,
local audio feature, or embedded file metadata using Mutagen, Spotipy, and Librosa.
Reads an iTunes Music Library XML (.xml), targets a playlist (optional), lets the user choose
an attribute and sort direction. Tries to read embedded metadata tags first; if not found,
falls back to Spotify lookup or Librosa analysis. Writes a new XML file with the playlist reordered.
"""
import sys
import subprocess
import os
import argparse
import xml.etree.ElementTree as ET
import datetime
import urllib.parse
import warnings

# suppress librosa/audioread and numpy warnings
warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mutagen import File as MutagenFile
import numpy as np
import librosa
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOauthError
from spotipy.exceptions import SpotifyException

# --- Dependency check ---
def check_dependencies():
    import importlib.util
    required = ['mutagen', 'librosa', 'numpy', 'spotipy']
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        resp = input(f"The following dependencies are missing: {', '.join(missing)}.\nInstall now? [y/N]: ").strip().lower()
        if resp == 'y':
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
                print("Dependencies installed. Please restart the script.")
            except Exception as e:
                print(f"Failed to install dependencies: {e}")
            sys.exit(0)
        else:
            print("Cannot proceed without dependencies. Exiting.")
            sys.exit(1)

check_dependencies()

# --- Embed your Spotify API credentials here ---
CLIENT_ID = 'XXXXXXXX'
CLIENT_SECRET = 'XXXXXXXX'

# Available attributes
ATTRIBUTES = [
    # Embedded/file metadata (tags)
    'title', 'artist', 'album', 'tracknumber', 'discnumber', 'genre',
    # Spotify metadata
    'popularity', 'release_date',
    # Local Librosa features
    'energy_local', 'brightness', 'percussiveness_zcr', 'percussiveness_onset',
    'contrast', 'style_and_key_similarity', 'bpm', 'music_genre',
    'harmonic_content_key', 'timbral_changes', 'dynamic_changes',
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
                    children = list(info)
                    for j in range(0, len(children), 2):
                        k = children[j].text
                        v = children[j+1].text
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
            return [list(item)[1].text for item in list(arr)
                    if list(item) and list(item)[0].text == 'Track ID']
    return []

def set_playlist_items(pl_dict, sorted_ids):
    # pl_dict is the <dict> for a single playlist
    for idx, elem in enumerate(list(pl_dict)):
        if elem.tag == 'key' and elem.text == 'Playlist Items':
            arr = list(pl_dict)[idx + 1]
            # clear out old entries
            for child in list(arr):
                arr.remove(child)
            # re-add in sorted order
            for tid in sorted_ids:
                d = ET.SubElement(arr, 'dict')
                ET.SubElement(d, 'key').text = 'Track ID'
                ET.SubElement(d, 'integer').text = tid
            return

def fetch_value(sp, artist, title, location, attr):
    """
    Fetch a single attribute: try embedded tags via Mutagen, then Spotify metadata, then Librosa analysis.
    """
    path = None
    if location:
        path = urllib.parse.unquote(location.replace('file://', ''))

    # 1) Embedded metadata
    if path:
        try:
            audio = MutagenFile(path, easy=True)
            if audio and audio.tags:
                tag_val = audio.tags.get(attr)
                if tag_val:
                    raw = tag_val[0]
                    if attr in ('tracknumber','discnumber'):
                        return int(raw.split('/')[0])
                    try:
                        return float(raw)
                    except ValueError:
                        return raw
        except Exception as e:
            print(f"[Metadata read error] {e}")

    # 2) Spotify metadata
    if attr in ('popularity','release_date'):
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
                return datetime.date.fromisoformat(rd) if '-' in rd else datetime.date(int(rd),1,1)
            return track.get(attr)
        except SpotifyException as e:
            print(f"[Spotify API Error] {e}")
            return None

    # 3) Local Librosa analysis
    if path and attr not in ('popularity','release_date'):
        try:
            y, sr = librosa.load(path, sr=None)
        except Exception as e:
            print(f"[Local analysis error] {e}")
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

def choose_attribute():
    print("Select a track attribute to sort by:")
    for i, a in enumerate(ATTRIBUTES, 1):
        print(f"  {i}. {a}")
    choice = input(f"Enter number or name (default 1 for '{ATTRIBUTES[0]}'): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(ATTRIBUTES):
        return ATTRIBUTES[int(choice)-1]
    if choice in ATTRIBUTES:
        return choice
    print(f"Invalid choice, defaulting to '{ATTRIBUTES[0]}'")
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

    parser = argparse.ArgumentParser(description='Sort an iTunes XML playlist by attribute or feature.')
    parser.add_argument('input',  help='Path to iTunes Music Library XML')
    parser.add_argument('playlist', nargs='?', help='Playlist name to sort')
    parser.add_argument('output',   nargs='?', help='Output XML path')
    args = parser.parse_args()

    tree, plist_dict, tracks_map = load_library_tree(args.input)
    playlists = list_playlists(plist_dict)
    pname = args.playlist or (playlists[0] if len(playlists) == 1 else None)
    if not pname:
        print(f"Available playlists: {playlists}")
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
    print(f"\nSorting by: {attr} ({dir_name})\n")

    try:
        auth = SpotifyClientCredentials(client_id=CLIENT_ID,
                                        client_secret=CLIENT_SECRET)
        sp = spotipy.Spotify(auth_manager=auth)
    except SpotifyOauthError as e:
        print(f"✗ Spotify authentication failed: {e}")
        sys.exit(1)

    scored = []
    print("Fetching values:")
    for tid in tids:
        info = tracks_map.get(tid, {})
        val = fetch_value(sp,
                          info.get('Artist'),
                          info.get('Name'),
                          info.get('Location'),
                          attr)
        if val is None:
            print(f"⚠️ '{info.get('Name')}' missing '{attr}', placing last.")
            val = datetime.date.min if attr=='release_date' else float('inf')
        print(f"  {info.get('Name')}: {attr} = {val}")
        scored.append((tid, val))

    # sort and rewrite playlist
    sorted_pairs = sorted(scored, key=lambda x: x[1], reverse=reverse)
    print("\nFinal order:")
    for i, (tid, v) in enumerate(sorted_pairs, 1):
        print(f"  {i}. {tracks_map[tid]['Name']} ({attr}={v})")

    sorted_ids = [tid for tid, _ in sorted_pairs]
    set_playlist_items(pl_dict, sorted_ids)

    # update playlist name
    for idx, elem in enumerate(list(plist_dict)):
        if elem.tag == 'key' and elem.text == 'Playlists':
            arr = list(plist_dict)[idx+1]
            for pl in arr:
                pts = list(pl)
                for j in range(len(pts)):
                    if pts[j].tag=='key' and pts[j].text=='Name' and pts[j+1].text==pname:
                        pts[j+1].text = f"{pname} : sorted by {attr}"
                        break
            break

    out = args.output or os.path.splitext(args.input)[0] + '_sorted.xml'
    tree.write(out, encoding='utf-8', xml_declaration=True)
    print(f"\n✓ '{pname}' sorted by '{attr}' ({dir_name}) saved to {out}")

if __name__ == '__main__':
    main()
