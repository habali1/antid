import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

export interface ResultCardProps {
  rank: number;
  species_name: string;
  common_name: string | null;
  taxon_id: number | null;
  similarity: number; // raw cosine similarity, 0.0 – 1.0 — NOT a probability
  geo_boosted: boolean;
  onPress: () => void;
}

export default function ResultCard({
  rank,
  species_name,
  common_name,
  similarity,
  geo_boosted,
  onPress,
}: ResultCardProps): React.JSX.Element {
  // Clamped defensively; the API contract is 0.0-1.0 but this is display code,
  // not validation, so it shouldn't crash on an out-of-range value.
  const score = Math.max(0, Math.min(1, similarity));
  const isTop = rank === 1;

  return (
    <Pressable
      style={({ pressed }) => [
        styles.card,
        isTop && styles.cardTop,
        pressed && { opacity: 0.9 },
      ]}
      onPress={onPress}
    >
      <View style={styles.row}>
        <Text style={[styles.rank, isTop && styles.rankTop]}>{rank}</Text>
        <View style={styles.names}>
          <View style={styles.nameRow}>
            <Text style={styles.species} numberOfLines={1}>
              {species_name}
            </Text>
            {geo_boosted && <Text style={styles.geoBadge}> 📍</Text>}
          </View>
          {!!common_name && (
            <Text style={styles.common} numberOfLines={1}>
              {common_name}
            </Text>
          )}
        </View>
      </View>

      <Text style={styles.scoreLabel}>Match score {score.toFixed(2)}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#1A1A1A',
    borderRadius: 16,
    padding: 16,
    marginBottom: 12,
  },
  cardTop: {
    backgroundColor: '#20231D',
    borderLeftWidth: 3,
    borderLeftColor: '#3DD68C',
  },
  row: { flexDirection: 'row', alignItems: 'center' },
  rank: { color: '#5A5A5A', fontSize: 32, fontWeight: '800', width: 44 },
  rankTop: { color: '#3DD68C' },
  names: { flex: 1, marginLeft: 8 },
  nameRow: { flexDirection: 'row', alignItems: 'center' },
  species: { color: '#FFFFFF', fontSize: 16, fontWeight: '700', flexShrink: 1 },
  geoBadge: { fontSize: 13 },
  common: { color: '#9A9A9A', fontSize: 13, marginTop: 2 },
  scoreLabel: { color: '#CFCFCF', fontSize: 12, textAlign: 'right', marginTop: 12 },
});
