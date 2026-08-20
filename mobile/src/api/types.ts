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
  };
  SpeciesDetail: {
    taxon_id: number | null;
    species_name: string;
    common_name: string | null;
  };
};
