/**
 * Registers "web" as a valid platform with the RN CLI so that
 * `react-native bundle/start --platform web` is accepted.
 * Module resolution for web is handled in metro.config.js
 * (bare 'react-native' → 'react-native-web').
 */
module.exports = {
  platforms: {
    web: {},
  },
};
