import { useEffect, useRef, useState } from 'react';
import { PermissionsAndroid, Platform } from 'react-native';
import Geolocation from 'react-native-geolocation-service';

export interface LocationState {
  lat: number | null;
  lon: number | null;
  permitted: boolean;
  loading: boolean;
}

const TIMEOUT_MS = 10000;

/**
 * Requests location permission once on mount and fetches a single fix.
 * On denial or timeout, resolves to (null, null, permitted:false) and never
 * re-requests. Location is always optional for the app flow.
 */
export function useLocation(): LocationState {
  const [state, setState] = useState<LocationState>({
    lat: null,
    lon: null,
    permitted: false,
    loading: true,
  });
  const done = useRef(false);

  useEffect(() => {
    let cancelled = false;

    const finishUnavailable = () => {
      if (!cancelled) {
        setState({ lat: null, lon: null, permitted: false, loading: false });
      }
    };

    const fetchFix = () => {
      Geolocation.getCurrentPosition(
        (pos) => {
          if (cancelled) return;
          setState({
            lat: pos.coords.latitude,
            lon: pos.coords.longitude,
            permitted: true,
            loading: false,
          });
        },
        () => finishUnavailable(),
        { enableHighAccuracy: false, timeout: TIMEOUT_MS, maximumAge: 60000 }
      );
    };

    const requestAndFetch = async () => {
      if (done.current) return;
      done.current = true;
      try {
        if (Platform.OS === 'ios') {
          const auth = await Geolocation.requestAuthorization('whenInUse');
          if (auth === 'granted') fetchFix();
          else finishUnavailable();
        } else if (Platform.OS === 'android') {
          const granted = await PermissionsAndroid.request(
            PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
            {
              title: 'Location permission',
              message: 'AntID uses your location to improve identification results.',
              buttonPositive: 'Allow',
              buttonNegative: 'Deny',
            }
          );
          if (granted === PermissionsAndroid.RESULTS.GRANTED) fetchFix();
          else finishUnavailable();
        } else if (Platform.OS === 'web') {
          // Web: spec mandates navigator.geolocation directly (no native module).
          if (typeof navigator !== 'undefined' && navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
              (pos) => {
                if (cancelled) return;
                setState({
                  lat: pos.coords.latitude,
                  lon: pos.coords.longitude,
                  permitted: true,
                  loading: false,
                });
              },
              () => finishUnavailable(),
              { enableHighAccuracy: false, timeout: TIMEOUT_MS, maximumAge: 60000 }
            );
          } else {
            finishUnavailable();
          }
        } else {
          finishUnavailable();
        }
      } catch {
        finishUnavailable();
      }
    };

    requestAndFetch();
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
