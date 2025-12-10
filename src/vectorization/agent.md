# Agent : Vectorization

## Mission

Convert textual and audio metadata into embeddings for similarity search.

## Inputs

- Clean text metadata
- Audio features extracted via Librosa

## Outputs

- Vector embeddings stored in Milvus

## Tools

- SentenceTransformers (text)
- Librosa (audio)
- Pymilvus (vector store)

## Notes

- Standardize vector sizes (normalize, same dimensions)
- Keep a mapping table: `music_id ↔ vector_id`
