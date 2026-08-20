module.exports = {
  // NativeWind's jsx preset stays off while all styling is StyleSheet-based;
  // re-add 'nativewind/babel' here when adopting className props.
  presets: ['module:@react-native/babel-preset'],
  plugins: [
    [
      'module-resolver',
      { root: ['.'], alias: { '@': './src' }, extensions: ['.ts', '.tsx', '.js', '.jsx', '.json'] },
    ],
    [
      'module:react-native-dotenv',
      { moduleName: '@env', path: '.env', allowUndefined: true },
    ],
    // Must remain LAST per reanimated docs.
    'react-native-reanimated/plugin',
  ],
};
