# Python-Music-Sorters

A collection of Python command‑line tools for sorting music collections and playlists based on embedded metadata, Spotify data, or local audio analysis using Librosa.

## Main Features

  * Reorder any playlist in your iTunes/Music.app library by a chosen attribute
  * Supports embedded metadata (BPM, key, mood, color)
  * Analyzes local audio features (energy, brightness, spectral contrast, onset strength, MFCC, etc.) via Librosa
  * Falls back to Spotify popularity if needed (currently deprecated by Spotify)

## Requirements

* Python 3.7 or higher
* [Mutagen](https://pypi.org/project/mutagen/) (ID3 & MP4 metadata parsing)
* [Librosa](https://pypi.org/project/librosa/) & [NumPy](https://pypi.org/project/numpy/) (audio feature extraction)
* [Spotipy](https://pypi.org/project/spotipy/) (Spotify Web API)

## Setup

1. Clone this repository:
2. Obtain Spotify API credentials (Client ID & Client Secret) and insert them at the top of the scripts:
   ```python
   CLIENT_ID = 'your_spotify_client_id'
   CLIENT_SECRET = 'your_spotify_client_secret'
````

## Usage Examples

* Follow the interactive prompts to choose a sort attribute and direction.

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes
4. Open a pull request

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
