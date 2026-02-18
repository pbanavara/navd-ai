/**
 * Compute cosine similarity between a query vector and a matrix of stored vectors.
 * Uses pre-computed norms for stored vectors to avoid redundant computation.
 * Returns an array of { index, score } sorted descending by score, limited to topK.
 */
export function topKSimilarity(
  query: Float32Array,
  vectors: Float32Array[],
  norms: number[],
  topK: number,
): Array<{ index: number; score: number }> {
  const results: Array<{ index: number; score: number }> = [];

  const queryNorm = norm(query);
  if (queryNorm === 0) return [];

  for (let i = 0; i < vectors.length; i++) {
    const vNorm = norms[i];
    if (vNorm === 0) continue;

    const v = vectors[i];
    let dot = 0;
    for (let j = 0; j < query.length; j++) {
      dot += query[j] * v[j];
    }
    const score = dot / (queryNorm * vNorm);
    results.push({ index: i, score });
  }

  results.sort((a, b) => b.score - a.score);
  return results.slice(0, topK);
}

function norm(v: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}
