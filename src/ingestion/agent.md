# Agent : Ingestion

## Mission

Collect and clean all metadata (artist, album, genre, year, description) from various sources.

## Inputs

- Raw text data from Guts of Darkness (HTML)
- Local music files (tags from Quodlibet or Mutagen)

## Outputs

- Clean JSON or CSV containing: title, artist, album, genre, description, path, date, duration

## Tools

- BeautifulSoup4 (scraping)
- Mutagen or Librosa (audio tags)
- Pandas (structuration)

## Notes

- Normalize encoding (UTF-8)
- Clean HTML tags and line breaks
- Use consistent field names (e.g. `artist_name`, `album_description`)
