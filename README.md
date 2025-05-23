# Python-Music-Sorters

A collection of Python command‑line tools for sorting music collections and playlists based on embedded metadata, Spotify data, or local audio analysis.

## Main Features

* **iTunes XML Playlist Sorter** (`sort_itunes_playlist.py`)

  * Reorder any playlist in your iTunes/Music.app library by a chosen attribute
  * Supports embedded metadata (BPM, key, mood, color)
  * Falls back to Spotify popularity if needed
  * Analyzes local audio features (energy, brightness, spectral contrast, onset strength, MFCC, etc.) via Librosa
  * Interactive, grouped CLI for intuitive attribute selection

* **Folder-Based MP3 Sorter** (`sort_folder_by_tag.py`)

  * Scan a directory of MP3 files and sort them into subfolders based on any ID3 tag (genre, artist, year, etc.)
  * Custom tag mappings via Mutagen

* **Extensible Architecture**

  * Modular `fetch_value()` function handling embedded tags, Spotify API calls, and Librosa feature extraction
  * Centralized attribute definitions and mapping for easy addition of new sort criteria
  * Common utility functions for XML parsing, playlist manipulation, and pretty‑printing

## Requirements

* Python 3.7 or higher
* [Mutagen](https://pypi.org/project/mutagen/) (ID3 & MP4 metadata parsing)
* [Librosa](https://pypi.org/project/librosa/) & [NumPy](https://pypi.org/project/numpy/) (audio feature extraction)
* [Spotipy](https://pypi.org/project/spotipy/) (Spotify Web API)

Install dependencies via:

```bash
pip install mutagen librosa numpy spotipy
```

## Setup

1. Clone this repository:

   ```bash
   ```

git clone [https://github.com/gianseo/Python-Music-Sorters.git](https://github.com/gianseo/Python-Music-Sorters.git)
cd Python-Music-Sorters

````
2. Obtain Spotify API credentials (Client ID & Client Secret) and insert them at the top of the scripts:
   ```python
   CLIENT_ID = 'your_spotify_client_id'
   CLIENT_SECRET = 'your_spotify_client_secret'
````

## Usage Examples

### Sort an iTunes XML Playlist

```bash
python sort_itunes_playlist.py ~/Music/iTunes\ Library.xml "My Playlist" my_playlist_sorted.xml
```

* Follow the interactive prompts to choose a sort attribute and direction.

### Sort MP3 Files in a Folder by Genre

```bash
python sort_folder_by_tag.py /path/to/mp3/folder genre
```

* Files will be moved into `genre/<GenreName>/` subfolders.

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes
4. Open a pull request

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
