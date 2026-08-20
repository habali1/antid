// Activates NativeWind / Tailwind on iOS & Android.
// Web resolves to ./nativewind.web.ts instead — vanilla Metro cannot parse
// raw CSS, and every component styles via StyleSheet, so web needs no css.
import '../../global.css';
