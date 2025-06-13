#!/usr/bin/env python3
"""
iTunes Music Library XML Playlist Sorter

Sort iTunes playlists by embedded metadata, Spotify popularity, or local audio analysis.
Supports ID3 tags (MP3) and MP4 tags (M4A/MP4) with comprehensive audio feature extraction.
"""

import sys
import subprocess
import os
import argparse
import xml.etree.ElementTree as ET
import urllib.parse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json

# Suppress audio processing warnings
warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global imports (checked later)
try:
    import numpy as np
    import librosa
    from mutagen import File as MutagenFile
    import mutagen.id3 as id3
    from mutagen.mp4 import MP4
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    from spotipy.exceptions import SpotifyException
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@dataclass
class SortAttribute:
    """Represents a sortable attribute with metadata."""
    key: str
    label: str
    description: str
    category: str
    requires_spotify: bool = False
    requires_audio: bool = False


@dataclass
class TrackInfo:
    """Represents track information from iTunes XML."""
    track_id: str
    name: Optional[str] = None
    artist: Optional[str] = None
    location: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        """Human-readable track identification."""
        parts = []
        if self.artist:
            parts.append(self.artist)
        if self.name:
            parts.append(self.name)
        return " - ".join(parts) if parts else f"Track {self.track_id}"


class Config:
    """Configuration and constants."""
    
    # Spotify credentials - set these or use environment variables
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'INSERTYOURSHERE')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', 'INSERTYOURSHERE')
    
    # Supported audio formats
    AUDIO_FORMATS = {'.mp3', '.m4a', '.mp4', '.m4b', '.flac', '.wav', '.aac'}
    MP4_FORMATS = {'.m4a', '.mp4', '.m4b'}
    
    # Embedded tag mappings
    EMBEDDED_TAG_MAP = {
        'beats_per_minute': ['TBPM'],
        'mood_danceability': ['MOOD_DANCEABILITY', 'DanceabilityAlgorithm'],
        'initial_key': ['TKEY', 'initialkey'],
        'mood_arousal': ['MOOD_AROUSAL'],
        'beatunes_color': ['beaTunes_COLOR'],
    }
    
    # Available sort attributes
    ATTRIBUTES = [
        # Embedded Metadata
        SortAttribute('beats_per_minute', 'Track BPM', 'Beats per minute from metadata', 'Embedded Metadata'),
        SortAttribute('mood_danceability', 'Danceability', 'Dance-ability rating', 'Embedded Metadata'),
        SortAttribute('initial_key', 'Musical Key', 'Key signature', 'Embedded Metadata'),
        SortAttribute('mood_arousal', 'Mood Arousal', 'Energy/arousal level', 'Embedded Metadata'),
        SortAttribute('beatunes_color', 'Color', 'beaTunes color classification', 'Embedded Metadata'),
        
        # Spotify Data
        SortAttribute('popularity', 'Spotify Popularity', 'Spotify popularity score (0-100)', 'Spotify', requires_spotify=True),
        
        # Local Audio Analysis
        SortAttribute('energy_local', 'RMS Energy', 'Root mean square energy', 'Local Analysis', requires_audio=True),
        SortAttribute('brightness', 'Spectral Centroid', 'Brightness/timbral centroid', 'Local Analysis', requires_audio=True),
        SortAttribute('percussiveness_zcr', 'Zero-Crossing Rate', 'Percussiveness indicator', 'Local Analysis', requires_audio=True),
        SortAttribute('percussiveness_onset', 'Onset Strength', 'Attack/onset detection', 'Local Analysis', requires_audio=True),
        SortAttribute('contrast', 'Spectral Contrast', 'Harmonic vs percussive content', 'Local Analysis', requires_audio=True),
        SortAttribute('style_and_key_similarity', 'Tonnetz', 'Tonal centroid features', 'Local Analysis', requires_audio=True),
        SortAttribute('bpm', 'Beat Tracker Tempo', 'Computed tempo', 'Local Analysis', requires_audio=True),
        SortAttribute('music_genre', 'Genre Proxy', 'MFCC-based genre estimation', 'Local Analysis', requires_audio=True),
        SortAttribute('harmonic_content_key', 'Harmonic Content', 'Chroma-based key detection', 'Local Analysis', requires_audio=True),
        SortAttribute('timbral_changes', 'Timbral Complexity', 'Spectral polynomial features', 'Local Analysis', requires_audio=True),
        SortAttribute('dynamic_changes', 'Dynamic Range', 'MFCC delta coefficients', 'Local Analysis', requires_audio=True),
    ]
    
    @classmethod
    def get_attribute(cls, key: str) -> Optional[SortAttribute]:
        """Get attribute by key."""
        return next((attr for attr in cls.ATTRIBUTES if attr.key == key), None)
    
    @classmethod
    def get_attributes_by_category(cls) -> Dict[str, List[SortAttribute]]:
        """Group attributes by category."""
        categories = {}
        for attr in cls.ATTRIBUTES:
            if attr.category not in categories:
                categories[attr.category] = []
            categories[attr.category].append(attr)
        return categories


class DependencyManager:
    """Handle dependency checking and installation."""
    
    @staticmethod
    def check_and_install():
        """Check for required dependencies and offer to install if missing."""
        if DEPENDENCIES_AVAILABLE:
            return True
            
        required = ['mutagen', 'librosa', 'numpy', 'spotipy']
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if not missing:
            return True
            
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("\nThese packages are required for the script to function.")
        
        if sys.stdin.isatty():  # Interactive terminal
            response = input("Install missing packages now? [y/N]: ").strip().lower()
            if response == 'y':
                try:
                    subprocess.check_call([
                        sys.executable, '-m', 'pip', 'install', '--user'
                    ] + missing)
                    print("\n✅ Packages installed successfully!")
                    print("Please restart the script to use the newly installed packages.")
                    return False
                except subprocess.CalledProcessError as e:
                    print(f"❌ Installation failed: {e}")
                    return False
        
        print(f"\nTo install manually, run:")
        print(f"  pip install {' '.join(missing)}")
        return False


class iTunesXMLHandler:
    """Handle iTunes XML parsing and manipulation."""
    
    def __init__(self, xml_path: str):
        self.xml_path = Path(xml_path)
        self.tree = None
        self.plist = None
        self.tracks = {}
        self._load_library()
    
    def _load_library(self):
        """Load and parse the iTunes XML library."""
        try:
            self.tree = ET.parse(self.xml_path)
            root = self.tree.getroot()
            self.plist = root.find('dict')
            
            if self.plist is None:
                raise ValueError("Invalid iTunes XML format: no plist dict found")
            
            self._parse_tracks()
            
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML file: {e}")
        except FileNotFoundError:
            raise ValueError(f"XML file not found: {self.xml_path}")
    
    def _parse_tracks(self):
        """Parse tracks from the XML."""
        plist_children = list(self.plist)
        
        for i, elem in enumerate(plist_children):
            if elem.tag == 'key' and elem.text == 'Tracks':
                if i + 1 < len(plist_children):
                    tracks_dict = plist_children[i + 1]
                    entries = list(tracks_dict)
                    
                    # Process key-value pairs
                    for key_elem, val_elem in zip(entries[::2], entries[1::2]):
                        track_id = key_elem.text
                        track_info = self._parse_track_info(val_elem)
                        track_info.track_id = track_id
                        self.tracks[track_id] = track_info
                break
    
    def _parse_track_info(self, track_dict) -> TrackInfo:
        """Parse individual track information."""
        info = TrackInfo("")
        children = list(track_dict)
        
        for key_elem, val_elem in zip(children[::2], children[1::2]):
            key = key_elem.text
            value = val_elem.text
            
            if key == 'Name':
                info.name = value
            elif key == 'Artist':
                info.artist = value
            elif key == 'Location':
                info.location = value
        
        return info
    
    def get_playlists(self) -> List[str]:
        """Get list of all playlist names."""
        playlists = []
        plist_children = list(self.plist)
        
        for i, elem in enumerate(plist_children):
            if elem.tag == 'key' and elem.text == 'Playlists':
                if i + 1 < len(plist_children):
                    playlists_array = plist_children[i + 1]
                    for playlist_dict in playlists_array:
                        playlist_children = list(playlist_dict)
                        for key_elem, val_elem in zip(playlist_children[::2], playlist_children[1::2]):
                            if key_elem.text == 'Name':
                                playlists.append(val_elem.text)
                                break
                break
        
        return playlists
    
    def get_playlist_tracks(self, playlist_name: str) -> List[str]:
        """Get track IDs for a specific playlist."""
        playlist_dict = self._find_playlist_dict(playlist_name)
        if not playlist_dict:
            return []
        
        playlist_children = list(playlist_dict)
        for i, elem in enumerate(playlist_children):
            if elem.tag == 'key' and elem.text == 'Playlist Items':
                if i + 1 < len(playlist_children):
                    items_array = playlist_children[i + 1]
                    track_ids = []
                    for item_dict in items_array:
                        integer_elem = item_dict.find('integer')
                        if integer_elem is not None:
                            track_ids.append(integer_elem.text)
                    return track_ids
        
        return []
    
    def _find_playlist_dict(self, playlist_name: str):
        """Find playlist dictionary by name."""
        plist_children = list(self.plist)
        
        for i, elem in enumerate(plist_children):
            if elem.tag == 'key' and elem.text == 'Playlists':
                if i + 1 < len(plist_children):
                    playlists_array = plist_children[i + 1]
                    for playlist_dict in playlists_array:
                        playlist_children = list(playlist_dict)
                        for key_elem, val_elem in zip(playlist_children[::2], playlist_children[1::2]):
                            if key_elem.text == 'Name' and val_elem.text == playlist_name:
                                return playlist_dict
        
        return None
    
    def update_playlist_order(self, playlist_name: str, sorted_track_ids: List[str]):
        """Update playlist with new track order."""
        playlist_dict = self._find_playlist_dict(playlist_name)
        if not playlist_dict:
            raise ValueError(f"Playlist '{playlist_name}' not found")
        
        playlist_children = list(playlist_dict)
        for i, elem in enumerate(playlist_children):
            if elem.tag == 'key' and elem.text == 'Playlist Items':
                if i + 1 < len(playlist_children):
                    items_array = playlist_children[i + 1]
                    
                    # Clear existing items
                    for child in list(items_array):
                        items_array.remove(child)
                    
                    # Add sorted items
                    for track_id in sorted_track_ids:
                        item_dict = ET.SubElement(items_array, 'dict')
                        ET.SubElement(item_dict, 'key').text = 'Track ID'
                        ET.SubElement(item_dict, 'integer').text = track_id
                
                break
    
    def rename_playlist(self, old_name: str, new_name: str):
        """Rename a playlist."""
        playlist_dict = self._find_playlist_dict(old_name)
        if not playlist_dict:
            return False
        
        playlist_children = list(playlist_dict)
        for key_elem, val_elem in zip(playlist_children[::2], playlist_children[1::2]):
            if key_elem.text == 'Name' and val_elem.text == old_name:
                val_elem.text = new_name
                return True
        
        return False
    
    def save(self, output_path: str):
        """Save the modified XML to file."""
        self._indent_xml(self.tree.getroot())
        self.tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def _indent_xml(self, elem, level=0):
        """Format XML with proper indentation."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                self._indent_xml(child, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class AudioAnalyzer:
    """Handle audio analysis and metadata extraction."""
    
    def __init__(self, spotify_client=None):
        self.spotify = spotify_client
    
    def get_track_value(self, track: TrackInfo, attribute: SortAttribute) -> Optional[float]:
        """Get the value for a specific attribute from a track."""
        file_path = self._get_file_path(track.location)
        
        # Try embedded metadata first
        if attribute.key in Config.EMBEDDED_TAG_MAP:
            value = self._get_embedded_value(file_path, attribute)
            if value is not None:
                return value
        
        # Try Spotify
        if attribute.requires_spotify and self.spotify:
            value = self._get_spotify_value(track, attribute)
            if value is not None:
                return value
        
        # Try local audio analysis
        if attribute.requires_audio and file_path:
            return self._get_audio_analysis_value(file_path, attribute)
        
        return None
    
    def _get_file_path(self, location: Optional[str]) -> Optional[str]:
        """Convert iTunes location to file path."""
        if not location:
            return None
        
        try:
            path = urllib.parse.unquote(location.replace('file://', ''))
            return path if os.path.exists(path) else None
        except Exception:
            return None
    
    def _get_embedded_value(self, file_path: Optional[str], attribute: SortAttribute) -> Optional[float]:
        """Extract embedded metadata from audio file."""
        if not file_path or not os.path.exists(file_path):
            return None
        
        ext = Path(file_path).suffix.lower()
        descriptors = Config.EMBEDDED_TAG_MAP.get(attribute.key, [])
        
        try:
            # MP4/M4A files
            if ext in Config.MP4_FORMATS:
                return self._get_mp4_value(file_path, descriptors, attribute)
            
            # ID3 (MP3) files
            else:
                return self._get_id3_value(file_path, descriptors, attribute)
        
        except Exception as e:
            print(f"⚠️  Error reading metadata from {Path(file_path).name}: {e}")
            return None
    
    def _get_mp4_value(self, file_path: str, descriptors: List[str], attribute: SortAttribute) -> Optional[float]:
        """Get value from MP4 tags."""
        try:
            mp4_file = MP4(file_path)
            tags = mp4_file.tags or {}
            
            # Special handling for BPM
            if attribute.key == 'beats_per_minute':
                bpm = tags.get('tmpo')
                if bpm:
                    return float(bpm[0])
            
            # Custom tags
            for desc in descriptors:
                key = f'----:com.apple.iTunes:{desc}'
                if key in tags:
                    raw = tags[key][0]
                    
                    # Special handling for mood arousal
                    if attribute.key == 'mood_arousal':
                        try:
                            raw_str = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else str(raw)
                            parts = raw_str.split(';')
                            return float(parts[1] if len(parts) > 1 else parts[0])
                        except Exception:
                            continue
                    
                    return self._convert_to_float(raw)
            
            return None
            
        except Exception:
            return None
    
    def _get_id3_value(self, file_path: str, descriptors: List[str], attribute: SortAttribute) -> Optional[float]:
        """Get value from ID3 tags."""
        try:
            id3_tags = id3.ID3(file_path)
            
            for desc in descriptors:
                # Standard frames
                for frame in id3_tags.getall(desc):
                    raw = getattr(frame, 'text', [None])[0] or getattr(frame, 'value', None)
                    if raw is not None:
                        return self._convert_to_float(raw)
                
                # TXXX frames
                for frame in id3_tags.getall('TXXX'):
                    if frame.desc == desc and frame.text:
                        return self._convert_to_float(frame.text[0])
            
            return None
            
        except Exception:
            return None
    
    def _get_spotify_value(self, track: TrackInfo, attribute: SortAttribute) -> Optional[float]:
        """Get value from Spotify API."""
        if attribute.key != 'popularity' or not self.spotify:
            return None
        
        try:
            # Build search query
            query_parts = []
            if track.artist:
                query_parts.append(f'artist:"{track.artist}"')
            if track.name:
                query_parts.append(f'track:"{track.name}"')
            
            query = ' '.join(query_parts) or track.name or track.artist
            if not query:
                return None
            
            # Search for track
            results = self.spotify.search(q=query, type='track', limit=1)
            tracks = results.get('tracks', {}).get('items', [])
            
            if tracks:
                track_id = tracks[0]['id']
                track_details = self.spotify.track(track_id)
                return float(track_details.get('popularity', 0))
            
            return None
            
        except SpotifyException as e:
            print(f"⚠️  Spotify error for {track.display_name}: {e}")
            return None
    
    def _get_audio_analysis_value(self, file_path: str, attribute: SortAttribute) -> Optional[float]:
        """Get value from local audio analysis."""
        try:
            y, sr = librosa.load(file_path, sr=None)
            
            if attribute.key == 'energy_local':
                return float(np.mean(librosa.feature.rms(y=y)))
            elif attribute.key == 'brightness':
                return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            elif attribute.key == 'percussiveness_zcr':
                return float(np.mean(librosa.feature.zero_crossing_rate(y)))
            elif attribute.key == 'percussiveness_onset':
                return float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
            elif attribute.key == 'contrast':
                return float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
            elif attribute.key == 'style_and_key_similarity':
                return float(np.mean(librosa.feature.tonnetz(y=y, sr=sr)))
            elif attribute.key == 'bpm':
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                return float(tempo)
            elif attribute.key == 'music_genre':
                return float(np.mean(librosa.feature.mfcc(y=y, sr=sr)))
            elif attribute.key == 'harmonic_content_key':
                return float(np.mean(librosa.feature.chroma_stft(y=y, sr=sr)))
            elif attribute.key == 'timbral_changes':
                return float(np.mean(librosa.feature.poly_features(y=y, sr=sr)))
            elif attribute.key == 'dynamic_changes':
                mfccs = librosa.feature.mfcc(y=y, sr=sr)
                return float(np.mean(librosa.feature.delta(mfccs)))
            
            return None
            
        except Exception as e:
            print(f"⚠️  Audio analysis error for {Path(file_path).name}: {e}")
            return None
    
    def _convert_to_float(self, value) -> Optional[float]:
        """Convert various value types to float."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            elif isinstance(value, (bytes, bytearray)):
                return float(value.decode('utf-8'))
            else:
                return float(str(value))
        except (ValueError, TypeError):
            return None


class PlaylistSorter:
    """Main application class."""
    
    def __init__(self):
        self.xml_handler = None
        self.audio_analyzer = None
        self.spotify_client = None
    
    def run(self):
        """Main application entry point."""
        if not DependencyManager.check_and_install():
            sys.exit(1)
        
        args = self._parse_arguments()
        
        try:
            # Load iTunes XML
            print(f"📚 Loading iTunes library: {args.input}")
            self.xml_handler = iTunesXMLHandler(args.input)
            
            # Setup Spotify if needed
            if self._needs_spotify(args):
                self._setup_spotify()
            
            # Setup audio analyzer
            self.audio_analyzer = AudioAnalyzer(self.spotify_client)
            
            # Process the playlist
            self._process_playlist(args)
            
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n\n👋 Cancelled by user")
            sys.exit(0)
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description='Sort iTunes XML playlists by various audio attributes',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_help_text()
        )
        
        parser.add_argument(
            'input',
            help='Path to iTunes Library XML file'
        )
        
        parser.add_argument(
            '-p', '--playlist',
            help='Playlist name to sort (will prompt if not provided)'
        )
        
        parser.add_argument(
            '-a', '--attribute',
            choices=[attr.key for attr in Config.ATTRIBUTES],
            help='Attribute to sort by (will prompt if not provided)'
        )
        
        parser.add_argument(
            '-d', '--descending',
            action='store_true',
            help='Sort in descending order (default: ascending)'
        )
        
        parser.add_argument(
            '-o', '--output',
            help='Output XML file path (default: input_sorted.xml)'
        )
        
        parser.add_argument(
            '--list-playlists',
            action='store_true',
            help='List all available playlists and exit'
        )
        
        parser.add_argument(
            '--list-attributes',
            action='store_true',
            help='List all available sort attributes and exit'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes'
        )
        
        parser.add_argument(
            '--no-rename',
            action='store_true',
            help='Don\'t rename the playlist to indicate sorting'
        )
        
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Show detailed progress information'
        )
        
        return parser.parse_args()
    
    def _get_help_text(self) -> str:
        """Generate help text with available attributes."""
        lines = ["\nAvailable sort attributes:"]
        
        categories = Config.get_attributes_by_category()
        for category, attributes in categories.items():
            lines.append(f"\n{category}:")
            for attr in attributes:
                req_text = ""
                if attr.requires_spotify:
                    req_text = " (requires Spotify)"
                elif attr.requires_audio:
                    req_text = " (requires audio analysis)"
                
                lines.append(f"  {attr.key:<25} {attr.description}{req_text}")
        
        lines.extend([
            "\nExamples:",
            "  %(prog)s library.xml --list-playlists",
            "  %(prog)s library.xml -p 'My Playlist' -a beats_per_minute",
            "  %(prog)s library.xml -p 'Dance Music' -a popularity --descending",
            "  %(prog)s library.xml --dry-run -a energy_local",
        ])
        
        return "\n".join(lines)
    
    def _needs_spotify(self, args) -> bool:
        """Check if Spotify credentials are needed."""
        if args.list_playlists or args.list_attributes:
            return False
        
        if args.attribute:
            attr = Config.get_attribute(args.attribute)
            return attr and attr.requires_spotify
        
        return True  # Might be needed during interactive selection
    
    def _setup_spotify(self):
        """Setup Spotify client."""
        if not Config.CLIENT_ID or not Config.CLIENT_SECRET:
            print("⚠️  Spotify credentials not configured.")
            print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
            print("or modify the CLIENT_ID and CLIENT_SECRET in the script.")
            return
        
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=Config.CLIENT_ID,
                client_secret=Config.CLIENT_SECRET
            )
            self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test the connection
            self.spotify_client.search(q="test", type="track", limit=1)
            print("🎵 Spotify connection established")
            
        except Exception as e:
            print(f"⚠️  Failed to connect to Spotify: {e}")
            self.spotify_client = None
    
    def _process_playlist(self, args):
        """Process the playlist sorting."""
        # Handle list commands
        if args.list_playlists:
            self._list_playlists()
            return

        if args.list_attributes:
            self._list_attributes()
            return

        # Get playlist
        playlist_name = self._get_playlist_name(args.playlist)
        track_ids = self.xml_handler.get_playlist_tracks(playlist_name)

        if not track_ids:
            print(f"❌ Playlist '{playlist_name}' is empty or not found")
            sys.exit(1)

        print(f"🎵 Found {len(track_ids)} tracks in '{playlist_name}'")

        # Get sort attribute
        attribute = self._get_sort_attribute(args.attribute)

        # Ask for sort direction
        while True:
            response = input("Choose sort direction: Ascending (A) or Descending (D) [A]: ").strip().lower()
            if response in ('', 'a', 'asc', 'ascending'):
                descending = False
                break
            elif response in ('d', 'desc', 'descending'):
                descending = True
                break
            else:
                print("Invalid input. Please enter 'A' for ascending or 'D' for descending.")

        # Sort tracks
        sorted_track_ids = self._sort_tracks(track_ids, attribute, descending, args.verbose)

        if args.dry_run:
            print("\n🔍 Dry run completed - no changes made")
            return

        # Update playlist
        self._update_playlist(playlist_name, sorted_track_ids, attribute, args)

        # Save results
        output_path = args.output or self._get_default_output_path(args.input)
        self._save_results(output_path, playlist_name, attribute)
    
    def _list_playlists(self):
        """List all available playlists."""
        playlists = self.xml_handler.get_playlists()
        
        if not playlists:
            print("No playlists found in the iTunes library")
            return
        
        print(f"\n📋 Found {len(playlists)} playlists:")
        for i, name in enumerate(playlists, 1):
            track_count = len(self.xml_handler.get_playlist_tracks(name))
            print(f"  {i:2d}. {name} ({track_count} tracks)")
    
    def _list_attributes(self):
        """List all available sort attributes."""
        print("\n🔧 Available sort attributes:")
        
        categories = Config.get_attributes_by_category()
        for category, attributes in categories.items():
            print(f"\n{category}:")
            for attr in attributes:
                req_text = ""
                if attr.requires_spotify:
                    req_text = " 🎵"
                elif attr.requires_audio:
                    req_text = " 🔊"
                
                print(f"  {attr.key:<25} {attr.description}{req_text}")
        
        print("\nLegend: 🎵 = Requires Spotify, 🔊 = Requires audio analysis")
    
    def _get_playlist_name(self, provided_name: Optional[str]) -> str:
        """Get playlist name from args or user input."""
        if provided_name:
            playlists = self.xml_handler.get_playlists()
            if provided_name not in playlists:
                print(f"❌ Playlist '{provided_name}' not found")
                print(f"Available playlists: {', '.join(playlists)}")
                sys.exit(1)
            return provided_name
        
        playlists = self.xml_handler.get_playlists()
        
        if not playlists:
            print("❌ No playlists found in the iTunes library")
            sys.exit(1)
        
        if len(playlists) == 1:
            print(f"📋 Using only available playlist: {playlists[0]}")
            return playlists[0]
        
        # Interactive selection
        print(f"\n📋 Available playlists:")
        for i, name in enumerate(playlists, 1):
            track_count = len(self.xml_handler.get_playlist_tracks(name))
            print(f"  {i:2d}. {name} ({track_count} tracks)")
        
        while True:
            try:
                choice = input(f"\nSelect playlist (1-{len(playlists)}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(playlists):
                        return playlists[idx]
                
                # Try exact match
                matches = [p for p in playlists if p.lower() == choice.lower()]
                if matches:
                    return matches[0]
                
                # Try partial match
                matches = [p for p in playlists if choice.lower() in p.lower()]
                if len(matches) == 1:
                    return matches[0]
                elif len(matches) > 1:
                    print(f"Multiple matches: {', '.join(matches)}")
                    continue
                
                print("Invalid selection. Try again.")
                
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Cancelled")
                sys.exit(0)
    
    def _get_sort_attribute(self, provided_attr: Optional[str]) -> SortAttribute:
        """Get sort attribute from args or user input."""
        if provided_attr:
            attr = Config.get_attribute(provided_attr)
            if not attr:
                print(f"❌ Unknown attribute: {provided_attr}")
                self._list_attributes()
                sys.exit(1)
            return attr
        
        # Interactive selection
        print("\n🔧 Select sort attribute:")
        
        categories = Config.get_attributes_by_category()
        all_attrs = []
        
        for category, attributes in categories.items():
            print(f"\n{category}:")
            for i, attr in enumerate(attributes):
                idx = len(all_attrs) + 1
                req_text = ""
                if attr.requires_spotify:
                    req_text = " 🎵" if self.spotify_client else " 🎵❌"
                elif attr.requires_audio:
                    req_text = " 🔊"
                
                print(f"  {idx:2d}. {attr.label}{req_text}")
                all_attrs.append(attr)
        
        print("\nLegend: 🎵 = Spotify required, 🔊 = Audio analysis, ❌ = Not available")
        
        while True:
            try:
                choice = input(f"\nSelect attribute (1-{len(all_attrs)}, default 1): ").strip()
                
                if not choice:
                    choice = "1"
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_attrs):
                        attr = all_attrs[idx]
                        
                        # Check availability
                        if attr.requires_spotify and not self.spotify_client:
                            print("❌ Spotify is required but not available for this attribute")
                            continue
                        
                        return attr
                
                print("Invalid selection. Try again.")
                
            except (ValueError, KeyboardInterrupt):
                print("\n👋 Cancelled")
                sys.exit(0)
    
    def _sort_tracks(self, track_ids: List[str], attribute: SortAttribute, 
                    descending: bool, verbose: bool) -> List[str]:
        """Sort tracks by the specified attribute."""
        print(f"\n🎯 Sorting by: {attribute.label}")
        if descending:
            print("📊 Direction: Descending (highest to lowest)")
        else:
            print("📊 Direction: Ascending (lowest to highest)")
        
        # Collect values
        scored_tracks = []
        missing_count = 0
        
        progress_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        
        for i, track_id in enumerate(track_ids):
            if not verbose and i % 10 == 0:
                char = progress_chars[i // 10 % len(progress_chars)]
                print(f"\r{char} Analyzing tracks... {i+1}/{len(track_ids)}", end="", flush=True)
            
            track = self.xml_handler.tracks.get(track_id)
            if not track:
                continue
            
            value = self.audio_analyzer.get_track_value(track, attribute)
            
            if value is None:
                if verbose:
                    print(f"⚠️  {track.display_name}: No {attribute.label} data")
                missing_count += 1
                value = float('inf')  # Sort missing values to end
            else:
                if verbose:
                    print(f"✅ {track.display_name}: {value:.3f}")
            
            scored_tracks.append((track_id, value, track))
        
        if not verbose:
            print("\r" + " " * 50 + "\r", end="")  # Clear progress line
        
        print(f"📈 Analysis complete: {len(scored_tracks)} tracks processed")
        if missing_count > 0:
            print(f"⚠️  {missing_count} tracks missing {attribute.label} data (will be sorted last)")
        
        # Sort tracks
        def sort_key(item):
            track_id, value, track = item
            # Sort missing values (inf) to the end regardless of direction
            if value == float('inf'):
                return (1, 0)  # Second sort group, neutral value
            else:
                return (0, value)  # First sort group, actual value
        
        sorted_tracks = sorted(scored_tracks, key=sort_key, reverse=descending)
        
        # Show preview
        print(f"\n📋 Sort preview (showing first 5 tracks):")
        for i, (track_id, value, track) in enumerate(sorted_tracks[:5]):
            value_str = "No data" if value == float('inf') else f"{value:.3f}"
            print(f"  {i+1:2d}. {track.display_name}: {value_str}")
        
        if len(sorted_tracks) > 5:
            print(f"  ... and {len(sorted_tracks) - 5} more tracks")
        
        return [track_id for track_id, _, _ in sorted_tracks]
    
    def _update_playlist(self, playlist_name: str, sorted_track_ids: List[str], 
                        attribute: SortAttribute, args):
        """Update the playlist with new track order."""
        # Update track order
        self.xml_handler.update_playlist_order(playlist_name, sorted_track_ids)
        
        # Rename playlist if requested
        if not args.no_rename:
            direction = "desc" if args.descending else "asc"
            new_name = f"{playlist_name} (sorted by {attribute.label}, {direction})"
            self.xml_handler.rename_playlist(playlist_name, new_name)
            print(f"📝 Playlist renamed to: {new_name}")
    
    def _get_default_output_path(self, input_path: str) -> str:
        """Generate default output path."""
        input_path = Path(input_path)
        return str(input_path.parent / f"{input_path.stem}_sorted{input_path.suffix}")
    
    def _save_results(self, output_path: str, playlist_name: str, attribute: SortAttribute):
        """Save the sorted XML and show results."""
        # Check if output file exists
        if os.path.exists(output_path):
            response = input(f"\n⚠️  File '{output_path}' already exists. Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                print("❌ Save cancelled")
                sys.exit(0)
        
        # Save file
        try:
            self.xml_handler.save(output_path)
            print(f"\n✅ Successfully saved sorted library to: {output_path}")
            print(f"🎵 Playlist '{playlist_name}' sorted by {attribute.label}")
            
            # Show file info
            file_size = os.path.getsize(output_path)
            print(f"📁 File size: {file_size:,} bytes")
            
        except Exception as e:
            print(f"❌ Failed to save file: {e}")
            sys.exit(1)


def main():
    """Main entry point."""
    try:
        app = PlaylistSorter()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
