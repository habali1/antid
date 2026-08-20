const { getDefaultConfig } = require('@react-native/metro-config');
const { withNativeWind } = require('nativewind/metro');

const config = getDefaultConfig(__dirname);

// Resolve web alongside native platforms (react-native-web via Metro).
config.resolver.platforms = ['ios', 'android', 'web', 'native'];

/**
 * Web resolution rules:
 *  1. Bare 'react-native' → 'react-native-web'.
 *  2. Deep imports into RN core internals (e.g. reanimated's
 *     `react-native/Libraries/StyleSheet/processColor`) reference files that
 *     are platform-split (.ios/.android) with no web variant. Fall back to
 *     the iOS variant for web — these are pure-JS helpers that only run
 *     behind native-platform guards in the browser.
 */
const defaultResolveRequest = config.resolver.resolveRequest;
const baseResolve = (context, moduleName, platform) =>
  (defaultResolveRequest || context.resolveRequest)(context, moduleName, platform);

config.resolver.resolveRequest = (context, moduleName, platform) => {
  // On web, stop Metro from falling back to `.native.*` files. Metro tries a
  // `.native` extension for EVERY platform (web included), and those variants
  // pull in TurboModule / native-only code that crashes react-native-web at
  // runtime — e.g. react-native-safe-area-context's InitialWindow.native.ts ->
  // TurboModuleRegistry (which RNW doesn't export). `.web.*` still wins.
  const ctx = platform === 'web' ? { ...context, preferNativePlatform: false } : context;

  if (platform === 'web' && moduleName === 'react-native') {
    return baseResolve(ctx, 'react-native-web', platform);
  }
  try {
    return baseResolve(ctx, moduleName, platform);
  } catch (e) {
    // Normalize backslashes so this path check works on Windows too, where
    // originModulePath uses '\' (otherwise the RN-core web fallback never fires).
    const origin = (context.originModulePath || '').replace(/\\/g, '/');
    const fromRNCore =
      platform === 'web' &&
      (origin.includes('node_modules/react-native/') ||
        moduleName.startsWith('react-native/'));
    if (fromRNCore) {
      return baseResolve(ctx, moduleName, 'ios');
    }
    throw e;
  }
};

module.exports = withNativeWind(config, { input: './global.css' });
