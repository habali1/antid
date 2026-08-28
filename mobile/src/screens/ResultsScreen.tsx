import React, { useCallback, useEffect, useState } from 'react';
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { CommonActions } from '@react-navigation/native';

import { RootStackParamList } from '@/api/types';
import ResultCard from '@/components/ResultCard';

type Props = NativeStackScreenProps<RootStackParamList, 'Results'>;

// The preview is sized to the photo's own aspect ratio, so a portrait shot
// stays portrait instead of being cropped to a wide banner. Clamped at both
// ends: an unclamped ratio lets a very tall photo push the results off screen
// and a panorama collapse to a sliver. `contain` means the clamped cases
// letterbox rather than crop, so the whole ant is always visible.
const MIN_ASPECT = 3 / 4; // tallest we render (portrait)
const MAX_ASPECT = 16 / 9; // widest we render (landscape)
// Portrait-friendly until the real ratio is measured — that's the common case
// here, and it keeps the first paint close to the final layout.
const DEFAULT_ASPECT = MIN_ASPECT;
// Never let the preview eat more than half the viewport.
const MAX_VIEWPORT_FRACTION = 0.5;

export default function ResultsScreen({
  route,
  navigation,
}: Props): React.JSX.Element {
  const {
    imageUri,
    results,
    inferenceMs,
    geoFiltered,
    gateActive,
    lowConfidence,
  } = route.params;
  const { height: windowHeight } = useWindowDimensions();
  const [aspect, setAspect] = useState(DEFAULT_ASPECT);

  useEffect(() => {
    let cancelled = false;
    Image.getSize(
      imageUri,
      (w, h) => {
        if (cancelled || !w || !h) return;
        setAspect(Math.min(MAX_ASPECT, Math.max(MIN_ASPECT, w / h)));
      },
      // Measurement can fail (revoked blob URL, remote fetch error). The
      // default ratio still renders the photo uncropped, so there is nothing
      // to recover from.
      () => undefined
    );
    return () => {
      cancelled = true;
    };
  }, [imageUri]);

  const tryAnother = useCallback(() => {
    navigation.dispatch(
      CommonActions.reset({ index: 0, routes: [{ name: 'Home' }] })
    );
  }, [navigation]);

  const openDetail = useCallback(
    (taxon_id: number | null, species_name: string, common_name: string | null) => {
      navigation.navigate('SpeciesDetail', { taxon_id, species_name, common_name });
    },
    [navigation]
  );

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Image
          source={{ uri: imageUri }}
          resizeMode="contain"
          style={[
            styles.thumb,
            {
              aspectRatio: aspect,
              maxHeight: windowHeight * MAX_VIEWPORT_FRACTION,
            },
          ]}
        />
        <Text style={styles.timing}>Compared in {inferenceMs}ms</Text>

        {geoFiltered && (
          <View style={styles.geoBanner}>
            <Text style={styles.geoBannerText}>
              📍 Location used to improve ranking
            </Text>
          </View>
        )}

        {gateActive === true && lowConfidence === true && (
          <View style={styles.lowConfidenceBanner}>
            <Text style={styles.lowConfidenceTitle}>No reliable match</Text>
            <Text style={styles.lowConfidenceText}>
              This may be a species outside the 50 supported species, or the
              photo may be too distant, blurry, or unclear. Try a closer,
              clearer photo. The results below are possible matches, not an
              identification.
            </Text>
          </View>
        )}

        <Text style={styles.sectionHeading}>Closest matches</Text>
        <Text style={styles.sectionCaption}>
          Compared with 50 supported species. Match scores show visual
          similarity, not the probability of a correct identification.
        </Text>

        {results.map((r) => (
          <ResultCard
            key={r.rank}
            rank={r.rank}
            species_name={r.species_name}
            common_name={r.common_name}
            taxon_id={r.taxon_id}
            similarity={r.similarity}
            geo_boosted={r.geo_boosted}
            onPress={() => openDetail(r.taxon_id, r.species_name, r.common_name)}
          />
        ))}
      </ScrollView>

      <Pressable
        style={({ pressed }) => [styles.button, pressed && { opacity: 0.85 }]}
        onPress={tryAnother}
      >
        <Text style={styles.buttonText}>Try Another</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D0D' },
  content: { padding: 20, paddingBottom: 12 },
  thumb: {
    width: '100%',
    borderRadius: 18,
    // Also the letterbox colour where the photo does not fill the box.
    backgroundColor: '#1A1A1A',
  },
  timing: { color: '#8A8A8A', fontSize: 13, marginTop: 12, marginBottom: 12 },
  sectionHeading: { color: '#FFFFFF', fontSize: 15, fontWeight: '700', marginBottom: 6 },
  sectionCaption: { color: '#8A8A8A', fontSize: 12.5, lineHeight: 17, marginBottom: 16 },
  geoBanner: {
    backgroundColor: '#15333A',
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 16,
  },
  geoBannerText: { color: '#7FD7E0', fontSize: 13, fontWeight: '500' },
  lowConfidenceBanner: {
    backgroundColor: '#332A16',
    borderColor: '#9C7930',
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 16,
  },
  lowConfidenceTitle: {
    color: '#F5D88B',
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 5,
  },
  lowConfidenceText: { color: '#E8DDBF', fontSize: 13, lineHeight: 18 },
  button: {
    backgroundColor: '#3DD68C',
    margin: 20,
    marginTop: 8,
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
  },
  buttonText: { color: '#0D0D0D', fontSize: 17, fontWeight: '700' },
});
