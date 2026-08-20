#!/usr/bin/env bash
# Generate the standard bare React Native ios/ and android/ projects for AntID
# and patch in the permissions this app needs. Run once after cloning:
#     npm run bootstrap:native
set -euo pipefail

RN_VERSION="0.76.5"
MOBILE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -d "$MOBILE_DIR/ios" && -d "$MOBILE_DIR/android" ]]; then
  echo "ios/ and android/ already exist — nothing to do."
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "Generating bare React Native $RN_VERSION native projects (this downloads the template)…"
(cd "$TMP" && npx --yes @react-native-community/cli@15 init AntID \
  --version "$RN_VERSION" --skip-install --pm npm --skip-git-init)

cp -R "$TMP/AntID/ios" "$MOBILE_DIR/ios"
cp -R "$TMP/AntID/android" "$MOBILE_DIR/android"

# sed -i differs between macOS (BSD) and Linux (GNU).
if [[ "$(uname)" == "Darwin" ]]; then SED=(sed -i ''); else SED=(sed -i); fi

# ---- iOS: usage-description strings required by image-picker + geolocation ----
PLIST="$MOBILE_DIR/ios/AntID/Info.plist"
add_plist() {
  grep -q "<key>$1</key>" "$PLIST" && return 0
  "${SED[@]}" "s#</dict>#\t<key>$1</key>\n\t<string>$2</string>\n</dict>#" "$PLIST"
}
add_plist NSCameraUsageDescription "AntID uses the camera to photograph ants for identification."
add_plist NSPhotoLibraryUsageDescription "AntID needs photo access so you can identify ants from your library."
add_plist NSLocationWhenInUseUsageDescription "AntID uses your location to show species found in your region."

# ---- Android: runtime permissions declared in the manifest ----
MANIFEST="$MOBILE_DIR/android/app/src/main/AndroidManifest.xml"
for PERM in android.permission.ACCESS_FINE_LOCATION android.permission.ACCESS_COARSE_LOCATION; do
  grep -q "$PERM" "$MANIFEST" || \
    "${SED[@]}" "s#<application#<uses-permission android:name=\"$PERM\" />\n    <application#" "$MANIFEST"
done

echo
echo "Done. Next steps:"
echo "  cd ios && pod install && cd .."
echo "  npx react-native run-ios      # or: npx react-native run-android"
