import React, { useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, StyleSheet, Text, View } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';

import { RootStackParamList, Species } from '@/api/types';
import { getSpecies } from '@/api/client';

type Props = NativeStackScreenProps<RootStackParamList, 'SpeciesDetail'>;

export default function SpeciesDetailScreen({
  route,
}: Props): React.JSX.Element {
  const { taxon_id, species_name, common_name } = route.params;
  const [match, setMatch] = useState<Species | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getSpecies()
      .then((list) => {
        if (cancelled) return;
        // Match on taxon_id only when we actually have one on both sides.
        // A null === null comparison matches the FIRST species in the list,
        // so every result would resolve to the same (wrong) species.
        setMatch(
          list.find((s) =>
            taxon_id !== null && s.taxon_id !== null
              ? s.taxon_id === taxon_id
              : s.species_name === species_name
          ) ?? null
        );
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [taxon_id, species_name]);

  // Prefer the freshly-fetched record, fall back to the params we were given.
  const displayName = match?.species_name ?? species_name;
  const displayCommon = match?.common_name ?? common_name;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.name}>{displayName}</Text>
      {!!displayCommon && <Text style={styles.common}>{displayCommon}</Text>}
      {taxon_id !== null && (
        <Text style={styles.taxon}>Taxon ID: {taxon_id}</Text>
      )}

      <View style={styles.divider} />

      <Text style={styles.heading}>About this species</Text>
      {loading ? (
        <ActivityIndicator color="#3DD68C" style={{ marginTop: 16 }} />
      ) : (
        <Text style={styles.body}>
          Detailed species information coming in a future update.
        </Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D0D' },
  content: { padding: 24 },
  name: { color: '#FFFFFF', fontSize: 26, fontWeight: '800' },
  common: { color: '#9A9A9A', fontSize: 16, marginTop: 6 },
  taxon: { color: '#6A6A6A', fontSize: 13, marginTop: 10 },
  divider: { height: 1, backgroundColor: '#262626', marginVertical: 24 },
  heading: { color: '#FFFFFF', fontSize: 18, fontWeight: '700' },
  body: { color: '#B0B0B0', fontSize: 15, marginTop: 12, lineHeight: 22 },
});
