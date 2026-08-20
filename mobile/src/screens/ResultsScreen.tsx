import React, { useCallback } from 'react';
import {
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { CommonActions } from '@react-navigation/native';

import { RootStackParamList } from '@/api/types';
import ResultCard from '@/components/ResultCard';

type Props = NativeStackScreenProps<RootStackParamList, 'Results'>;

export default function ResultsScreen({
  route,
  navigation,
}: Props): React.JSX.Element {
  const { imageUri, results, inferenceMs, geoFiltered } = route.params;

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
        <Image source={{ uri: imageUri }} style={styles.thumb} />
        <Text style={styles.timing}>Identified in {inferenceMs}ms</Text>

        {geoFiltered && (
          <View style={styles.geoBanner}>
            <Text style={styles.geoBannerText}>
              📍 Results filtered for your region
            </Text>
          </View>
        )}

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
    height: 220,
    borderRadius: 18,
    backgroundColor: '#1A1A1A',
    resizeMode: 'cover',
  },
  timing: { color: '#8A8A8A', fontSize: 13, marginTop: 12, marginBottom: 12 },
  geoBanner: {
    backgroundColor: '#15333A',
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 16,
  },
  geoBannerText: { color: '#7FD7E0', fontSize: 13, fontWeight: '500' },
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
