// common_name and taxon_id are nullable on the wire: the API serves whatever
// is in taxonomy.json, and a training run that had no taxonomic metadata to
// work with emits null for both. Screens must degrade rather than assume.
export interface IdentificationResult {
  rank: number;
  species_name: string;
  common_name: string | null;
  taxon_id: number | null;
  similarity: number;
  geo_boosted: boolean;
}

export interface IdentificationResponse {
  results: IdentificationResult[];
  inference_ms: number;
  geo_filtered: boolean;
  // Optional during rollout: an older API may omit both. LoadingScreen
  // normalizes an absent or malformed gate response to inactive/null so the
  // client never implies that an unverified match is reliable.
  gate_active?: boolean;
  low_confidence?: boolean | null;
}

export interface Species {
  species_name: string;
  common_name: string | null;
  taxon_id: number | null;
  slug: string;
}

export type RootStackParamList = {
  Home: undefined;
  Loading: { imageUri: string; lat: number | null; lon: number | null };
  Results: {
    imageUri: string;
    results: IdentificationResult[];
    inferenceMs: number;
    geoFiltered: boolean;
    gateActive: boolean;
    lowConfidence: boolean | null;
  };
  SpeciesDetail: {
    taxon_id: number | null;
    species_name: string;
    common_name: string | null;
  };
};
