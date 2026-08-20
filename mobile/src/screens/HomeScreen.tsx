import React, { useCallback, useRef } from 'react';
import {
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import {
  launchCamera,
  launchImageLibrary,
  ImagePickerResponse,
} from 'react-native-image-picker';

import { RootStackParamList } from '@/api/types';
import { useLocation } from '@/hooks/useLocation';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

export default function HomeScreen({ navigation }: Props): React.JSX.Element {
  // Pre-fetch GPS while the user is still choosing a photo.
  const { lat, lon, permitted, loading } = useLocation();

  const goLoading = useCallback(
    (uri: string) => {
      navigation.navigate('Loading', { imageUri: uri, lat, lon });
    },
    [navigation, lat, lon]
  );

  const handlePicker = useCallback(
    (res: ImagePickerResponse) => {
      if (res.didCancel) return;
      const uri = res.assets?.[0]?.uri;
      if (uri) goLoading(uri);
    },
    [goLoading]
  );

  const onTakePhoto = useCallback(async () => {
    const res = await launchCamera({ mediaType: 'photo', quality: 0.8 });
    handlePicker(res);
  }, [handlePicker]);

  const onChooseLibrary = useCallback(async () => {
    const res = await launchImageLibrary({ mediaType: 'photo', quality: 0.8 });
    handlePicker(res);
  }, [handlePicker]);

  // ---- web file input ----
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const onWebFile = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const uri = URL.createObjectURL(file);
      // Stash the File so client.ts can upload the real bytes on web.
      (globalThis as { __antidWebFile?: File }).__antidWebFile = file;
      goLoading(uri);
    },
    [goLoading]
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>AntID</Text>
        <Text style={styles.subtitle}>Identify an ant from a photo</Text>
      </View>

      <View style={styles.buttons}>
        {Platform.OS === 'web' ? (
          <>
            {/* Rendered as a styled label wrapping a hidden file input. */}
            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
              onPress={() => fileInputRef.current?.click()}
            >
              <Text style={styles.buttonText}>Take or Choose Photo</Text>
            </Pressable>
            {/* DOM intrinsic — rendered only when Platform.OS === 'web' */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              style={{ display: 'none' }}
              onChange={onWebFile}
            />
          </>
        ) : (
          <>
            <Pressable
              style={({ pressed }) => [styles.button, pressed && styles.buttonPressed]}
              onPress={onTakePhoto}
            >
              <Text style={styles.buttonText}>Take Photo</Text>
            </Pressable>
            <Pressable
              style={({ pressed }) => [
                styles.button,
                styles.buttonSecondary,
                pressed && styles.buttonPressed,
              ]}
              onPress={onChooseLibrary}
            >
              <Text style={[styles.buttonText, styles.buttonTextLight]}>Choose from Library</Text>
            </Pressable>
          </>
        )}
      </View>

      <LocationStatus permitted={permitted} loading={loading} />
    </View>
  );
}

function LocationStatus({
  permitted,
  loading,
}: {
  permitted: boolean;
  loading: boolean;
}): React.JSX.Element {
  let dotColor = '#666';
  let label = 'Location unavailable — enable for better results';
  if (loading) {
    label = 'Getting location…';
  } else if (permitted) {
    dotColor = '#3DD68C';
    label = '📍 Location active — results will be filtered for your region';
  }
  return (
    <View style={styles.locationRow}>
      <View style={[styles.dot, { backgroundColor: dotColor }]} />
      <Text style={styles.locationText}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D0D', padding: 24, justifyContent: 'space-between' },
  header: { marginTop: 48, alignItems: 'center' },
  title: { color: '#FFFFFF', fontSize: 40, fontWeight: '800', letterSpacing: 1 },
  subtitle: { color: '#9A9A9A', fontSize: 15, marginTop: 8 },
  buttons: { gap: 16 },
  button: {
    backgroundColor: '#3DD68C',
    paddingVertical: 18,
    borderRadius: 16,
    alignItems: 'center',
  },
  buttonSecondary: { backgroundColor: '#1A1A1A', borderWidth: 1, borderColor: '#2A2A2A' },
  buttonPressed: { opacity: 0.85 },
  buttonText: { color: '#0D0D0D', fontSize: 17, fontWeight: '700' },
  buttonTextLight: { color: '#FFFFFF' },
  locationRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginBottom: 24 },
  dot: { width: 9, height: 9, borderRadius: 5, marginRight: 8 },
  locationText: { color: '#9A9A9A', fontSize: 13, textAlign: 'center', flexShrink: 1 },
});
