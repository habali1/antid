import React, { useEffect, useRef } from 'react';
import { Animated, Easing, Pressable, StyleSheet, Text, View } from 'react-native';

export interface ResultCardProps {
  rank: number;
  species_name: string;
  common_name: string | null;
  taxon_id: number | null;
  similarity: number; // 0.0 – 1.0
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
  const widthAnim = useRef(new Animated.Value(0)).current;
  const pct = Math.max(0, Math.min(1, similarity));

  useEffect(() => {
    Animated.timing(widthAnim, {
      toValue: pct,
      duration: 600,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false, // animating width
    }).start();
  }, [pct, widthAnim]);

  const fillWidth = widthAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

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

      <Text style={styles.pctLabel}>{Math.round(pct * 100)}%</Text>
      <View style={styles.track}>
        <Animated.View
          style={[styles.fill, isTop && styles.fillTop, { width: fillWidth }]}
        />
      </View>
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
  pctLabel: { color: '#CFCFCF', fontSize: 12, textAlign: 'right', marginTop: 12, marginBottom: 4 },
  track: { height: 8, backgroundColor: '#2C2C2C', borderRadius: 4, overflow: 'hidden' },
  fill: { height: '100%', backgroundColor: '#6C7A89', borderRadius: 4 },
  fillTop: { backgroundColor: '#3DD68C' },
});
