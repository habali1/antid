# AntID — mobile app

Bare React Native **0.76.5** (TypeScript) targeting **iOS, Android, and Web**
(react-native-web served by Metro). Photo → `POST /identify` → ranked results,
with optional GPS-based geo re-ranking.

## Setup

```bash
npm install
cp .env.example .env        # set API_BASE_URL (default http://localhost:8000)
```

`.env` is read at build time (react-native-dotenv). Android emulators reach
your host machine via `http://10.0.2.2:8000` — if `API_BASE_URL` is unset the
app falls back to that automatically on Android.

## Run

| Target | Command | Notes |
|---|---|---|
| iOS | `cd ios && pod install && cd .. && npm run ios` | macOS + Xcode |
| Android | `npm run android` | emulator or device |
| Web | `npm run web` → open **http://localhost:3000** | starts Metro + a static page server |

`npm run typecheck` runs strict `tsc` over the whole app.

## How the pieces fit

```
App.tsx                     native-stack: Home → Loading → Results → SpeciesDetail
src/hooks/useLocation.ts    one-shot permission + 10s GPS fix (iOS/Android/web), optional
src/api/client.ts           axios; multipart upload; lat/lon as query params when both set
src/api/types.ts            API contract types + RootStackParamList
src/screens/*               UI (dark #0D0D0D, StyleSheet-based)
src/components/ResultCard   rank card with 600ms animated similarity bar + 📍 geo badge
```

Location is always optional: every screen works identically with
`lat: null, lon: null` (denied permission, timeout, or no fix).

On **web**, the camera/library buttons collapse to a single styled
`<input type="file" accept="image/*" capture="environment">`, and the chosen
`File`'s bytes are uploaded directly.

## Platform / config notes

- **NativeWind v4 is wired but dormant.** All components use
  `StyleSheet.create` (the spec's fallback). The tailwind config, `global.css`,
  and Metro integration are in place — to adopt `className` props, re-add
  `'nativewind/babel'` to `presets` in `babel.config.js`. It's currently off
  because its runtime drags reanimated's RN-core deep imports into the web
  bundle.
- **Web resolution** lives in `metro.config.js`: bare `react-native` maps to
  `react-native-web`, and RN-core platform-split internals fall back to their
  iOS (pure-JS) variants. `react-native.config.js` registers `web` as a valid
  CLI platform.
- **Permissions** are already declared: iOS `Info.plist` has camera, photo
  library, and when-in-use location strings; the Android manifest declares
  fine/coarse location only (the camera launches via the system intent, which
  needs no declared permission).
- `scripts/bootstrap_native.sh` regenerates `ios/` + `android/` from the RN
  0.76.5 template if they're ever deleted; the checked-in folders already
  include the permission patches.

## Verified in CI-like conditions

`npm install`, strict `tsc --noEmit`, full Metro release bundles for **ios**
and **web**, and `npm run web` serving the page + live web bundle were all run
green. Launching the iOS simulator / Android emulator / a real browser
requires your machine.
