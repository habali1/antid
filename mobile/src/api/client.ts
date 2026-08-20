import axios, { AxiosInstance } from 'axios';
import { Platform } from 'react-native';
import { API_BASE_URL } from './env';
import { IdentificationResponse } from './types';

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

/** Build the multipart body. On web, imageUri is an object URL backed by a File. */
function buildFormData(
  imageUri: string,
  webFile?: File | null
): FormData {
  const form = new FormData();
  if (Platform.OS === 'web') {
    if (webFile) {
      form.append('file', webFile, webFile.name || 'photo.jpg');
    } else {
      // Fallback: fetch the object URL into a blob (handled by caller normally).
      throw new ApiError('No file provided for web upload.');
    }
  } else {
    // React Native multipart file descriptor.
    // @ts-expect-error RN FormData accepts this shape at runtime.
    form.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'photo.jpg',
    });
  }
  return form;
}

/**
 * Identify an ant from an image, optionally re-ranked by location.
 * lat/lon are appended as query params only when BOTH are non-null.
 */
export async function identifyImage(
  imageUri: string,
  lat: number | null,
  lon: number | null,
  webFile?: File | null
): Promise<IdentificationResponse> {
  const form = buildFormData(imageUri, webFile);
  const params: Record<string, number> = {};
  if (lat !== null && lon !== null) {
    params.lat = lat;
    params.lon = lon;
  }

  try {
    const res = await api.post<IdentificationResponse>('/identify', form, {
      params,
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  } catch (err) {
    if (axios.isAxiosError(err)) {
      if (err.response) {
        const status = err.response.status;
        if (status === 422) {
          throw new ApiError(
            "That image couldn't be processed. Try a clearer photo of a single ant.",
            422
          );
        }
        if (status === 503) {
          throw new ApiError('The identification service is starting up. Try again shortly.', 503);
        }
        const detail =
          (err.response.data as { detail?: string } | undefined)?.detail;
        throw new ApiError(detail || `Server error (${status}).`, status);
      }
      throw new ApiError(
        "Couldn't reach the server. Check your connection and that the API is running."
      );
    }
    throw new ApiError('Something went wrong while identifying the image.');
  }
}

/** GET /health → true when the service reports status "ok". */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await api.get<{ status: string }>('/health', { timeout: 5000 });
    return res.data?.status === 'ok';
  } catch {
    return false;
  }
}

/** GET /species → list of all known species objects (used by SpeciesDetail). */
export async function getSpecies(): Promise<
  import('./types').Species[]
> {
  const res = await api.get<{ species: import('./types').Species[] }>('/species');
  return res.data.species;
}
