import '@/styles/nativewind';

import React from 'react';
import { StatusBar } from 'react-native';
import { DarkTheme, NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import { RootStackParamList } from '@/api/types';
import HomeScreen from '@/screens/HomeScreen';
import LoadingScreen from '@/screens/LoadingScreen';
import ResultsScreen from '@/screens/ResultsScreen';
import SpeciesDetailScreen from '@/screens/SpeciesDetailScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

const theme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: '#0D0D0D',
    card: '#0D0D0D',
    text: '#FFFFFF',
    primary: '#3DD68C',
    border: '#262626',
  },
};

export default function App(): React.JSX.Element {
  return (
    <NavigationContainer theme={theme}>
      <StatusBar barStyle="light-content" backgroundColor="#0D0D0D" />
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: { backgroundColor: '#0D0D0D' },
          headerTintColor: '#FFFFFF',
          headerShadowVisible: false,
        }}
      >
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ headerShown: false }}
        />
        <Stack.Screen
          name="Loading"
          component={LoadingScreen}
          // No header & no swipe-back: the request must not be interrupted.
          options={{ headerShown: false, gestureEnabled: false }}
        />
        <Stack.Screen
          name="Results"
          component={ResultsScreen}
          options={{ headerShown: false, gestureEnabled: false }}
        />
        <Stack.Screen
          name="SpeciesDetail"
          component={SpeciesDetailScreen}
          // Header shown: provides the back button to ResultsScreen.
          options={{ title: 'Species' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
