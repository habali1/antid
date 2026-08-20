/**
 * AntID entry point — registers the app for native and mounts it on web.
 */
import { AppRegistry, Platform } from 'react-native';
import App from './App';
import { name as appName } from './app.json';

AppRegistry.registerComponent(appName, () => App);

if (Platform.OS === 'web') {
  const mount = () =>
    AppRegistry.runApplication(appName, {
      rootTag: document.getElementById('root'),
    });
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
}
