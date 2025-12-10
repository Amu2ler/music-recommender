# Agent : Recommendation

## Mission

Compute and rank music recommendations based on similarity and user preferences.

## Inputs

- Query vector (music or text)
- User interactions (likes, history)
- Milvus search results

## Outputs

- Ranked list of recommended music items
- Optional explanation of recommendations (RAG-style)

## Tools

- Scikit-learn (similarity metrics)
- Milvus
- FastAPI for API endpoints

## Notes

- Implement hybrid scoring: 0.5 _ sim_text + 0.5 _ sim_audio
- Add user preference weighting later
