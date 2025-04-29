#!/usr/bin/env python3
"""
analyze_audio.py

Estimate tempo (BPM) and RMS energy percentage for each audio file,
write BPM only if missing, and append an
“Energy Level : XX%” line to the comments tag—unless one is already present.

Supported formats:
  • MP3/ID3   – TBPM (BPM), COMM (comments)
  • FLAC      – TBPM (BPM), COMMENT (Vorbis comment)
  • M4A/MP4   – tmpo (BPM), ©cmt (comments)
  • Fallback  – TBPM & COMMENT tags if available

Dependencies:
    pip3 install librosa mutagen
"""

import sys
import os
import traceback
import librosa

from mutagen import File
from mutagen.id3 import ID3, TBPM, COMM, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.mp4 import MP4

def percent(rms: float) -> int:
    """Convert 0.0–1.0 RMS to an integer percentage 0–100."""
    return int(round(rms * 100))

def append_energy_comment(existing_lines, pct):
    """
    Return a list of comment lines with "Energy Level : XX%" appended
    only if not already present.
    """
    # If any existing line already begins with "Energy Level :"
    if any(line.startswith("Energy Level :") for line in existing_lines):
        return existing_lines
    return existing_lines + [f"Energy Level : {pct}%"]

def set_id3_bpm_and_comment(path, bpm, rms, wrote_bpm):
    """Write BPM (if requested) and append Energy Level comment to an MP3."""
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()

    # BPM
    if wrote_bpm:
        tags.add(TBPM(encoding=3, text=str(int(round(bpm)))))

    # Comments
    pct = percent(rms)
    existing = []
    for comm in tags.getall('COMM'):
        existing.extend(comm.text)
    new_comments = append_energy_comment(existing, pct)

    tags.delall('COMM')
    tags.add(COMM(encoding=3, lang='eng', desc='', text=new_comments))
    tags.save(path)

def set_flac_bpm_and_comment(path, bpm, rms, wrote_bpm):
    """Write BPM (if requested) and append Energy Level comment to a FLAC."""
    audio = FLAC(path)
    if wrote_bpm:
        audio['TBPM'] = str(int(round(bpm)))

    pct = percent(rms)
    existing = audio.get('COMMENT', [])
    new_comments = append_energy_comment(existing, pct)
    audio['COMMENT'] = new_comments
    audio.save()

def set_mp4_bpm_and_comment(path, bpm, rms, wrote_bpm):
    """Write BPM (if requested) and append Energy Level comment to an MP4/M4A."""
    audio = MP4(path)
    if wrote_bpm:
        audio.tags['tmpo'] = [int(round(bpm))]

    key = '\u00A9cmt'
    existing = audio.tags.get(key, [])
    new_comments = append_energy_comment(existing, percent(rms))
    audio.tags[key] = new_comments
    audio.save()

def process(path):
    """Load audio, estimate tempo and RMS%, then dispatch to tag-writers."""
    # load & analyze
    y, sr = librosa.load(path, sr=None, mono=True)
    raw_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(raw_tempo)
    mean_rms = float(librosa.feature.rms(y=y).mean())

    # detect existing BPM
    ext = os.path.splitext(path)[1].lower()
    if ext == '.mp3':
        try:
            tags = ID3(path)
            has_bpm = bool(tags.getall('TBPM'))
        except ID3NoHeaderError:
            has_bpm = False
        set_id3_bpm_and_comment(path, tempo, mean_rms, not has_bpm)

    elif ext == '.flac':
        audio = FLAC(path)
        has_bpm = 'TBPM' in audio
        set_flac_bpm_and_comment(path, tempo, mean_rms, not has_bpm)

    elif ext in ('.m4a', '.mp4'):
        audio = MP4(path)
        has_bpm = 'tmpo' in audio.tags
        set_mp4_bpm_and_comment(path, tempo, mean_rms, not has_bpm)

    else:
        # generic fallback
        audio = File(path)
        if audio and audio.tags is not None:
            has_bpm = 'TBPM' in audio.tags
            if not has_bpm:
                audio.tags['TBPM'] = str(int(round(tempo)))
            key = 'COMMENT'
            existing = audio.tags.get(key, [])
            new_comments = append_energy_comment(existing, percent(mean_rms))
            audio.tags[key] = new_comments
            audio.save()
        else:
            has_bpm = False

    # print status
    bpm_msg = f"{tempo:.1f}" if not has_bpm else "(kept)"
    print(f"[OK] {os.path.basename(path)} → BPM={bpm_msg}; Energy Level={percent(mean_rms)}%")

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_audio.py <file1> [file2 ...]", file=sys.stderr)
        sys.exit(1)
    for f in sys.argv[1:]:
        try:
            process(f)
        except Exception as e:
            print(f"[ERROR] {f}: {e}", file=sys.stderr)
            traceback.print_exc()

if __name__ == '__main__':
    main()
