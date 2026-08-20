import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  BackHandler,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import { RootStackParamList } from '@/api/types';
import { identifyImage, ApiError } from '@/api/client';

type Props = NativeStackScreenProps<RootStackParamList, 'Loading'>;

export default function LoadingScreen({
  route,
  navigation,
}: Props): React.JSX.Element {
  const { imageUri, lat, lon } = route.params;
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  // Block hardware back / swipe while a request is in flight. Native only:
  // BackHandler is unsupported on web (RNW just logs an error), and there is no
  // hardware back button in a browser, so the guard is a no-op there.
  useEffect(() => {
    if (Platform.OS === 'web') return;
    const sub = BackHandler.addEventListener('hardwareBackPress', () => true);
    return () => sub.remove();
  }, []);

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    let cancelled = false;
    const webFile =
      Platform.OS === 'web'
        ? (globalThis as { __antidWebFile?: File }).__antidWebFile ?? null
        : null;

    identifyImage(imageUri, lat, lon, webFile)
      .then((resp) => {
        if (cancelled) return;
        navigation.replace('Results', {
          imageUri,
          results: resp.results,
          inferenceMs: resp.inference_ms,
          geoFiltered: resp.geo_filtered,
        });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg =
          err instanceof ApiError
            ? err.message
            : 'Something went wrong. Please try again.';
        setError(msg);
      });

    return () => {
      cancelled = true;
    };
  }, [imageUri, lat, lon, navigation]);

  const hasLocation = lat !== null && lon !== null;

  if (error) {
    return (
      <View style={styles.container}>
        <Image source={{ uri: imageUri }} style={styles.thumb} />
        <Text style={styles.errorText}>{error}</Text>
        <Pressable
          style={({ pressed }) => [styles.button, pressed && { opacity: 0.85 }]}
          onPress={() => navigation.navigate('Home')}
        >
          <Text style={styles.buttonText}>Try Again</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Image source={{ uri: imageUri }} style={styles.thumb} />
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#3DD68C" />
        <Text style={styles.identifying}>Identifying…</Text>
        <Text style={styles.subtle}>
          {hasLocation ? '📍 Using your location' : 'No location data'}
        </Text>
      </View>
      <View style={{ height: 60 }} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D0D', alignItems: 'center', padding: 24 },
  thumb: {
    width: '100%',
    height: '40%',
    borderRadius: 18,
    marginTop: 12,
    backgroundColor: '#1A1A1A',
    resizeMode: 'cover',
  },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  identifying: { color: '#FFFFFF', fontSize: 20, fontWeight: '600', marginTop: 20 },
  subtle: { color: '#7A7A7A', fontSize: 14, marginTop: 8 },
  errorText: { color: '#FF6B6B', fontSize: 16, textAlign: 'center', marginVertical: 28, paddingHorizontal: 12 },
  button: { backgroundColor: '#3DD68C', paddingVertical: 16, paddingHorizontal: 48, borderRadius: 16 },
  buttonText: { color: '#0D0D0D', fontSize: 17, fontWeight: '700' },
});
