import { Platform } from 'react-native';
// Injected at build time by babel (react-native-dotenv) from mobile/.env —
// copy .env.example to .env and edit. Works identically on iOS/Android/web.
import { API_BASE_URL as ENV_API_BASE_URL } from '@env';

/**
 * Resolve the API base URL. Priority:
 *   1. API_BASE_URL from .env (all platforms)
 *   2. Dev fallback — Android emulators reach the host machine via 10.0.2.2.
 */
function resolveBaseUrl(): string {
  if (ENV_API_BASE_URL) {
    return ENV_API_BASE_URL;
  }
  return Platform.OS === 'android'
    ? 'http://10.0.2.2:8000'
    : 'http://localhost:8000';
}

export const API_BASE_URL = resolveBaseUrl();
