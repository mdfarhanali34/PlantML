import React from 'react';
import { View } from 'react-native';
import { CameraScreen } from 'react-native-camera-kit';

const CustomCameraScreen = () => {
  return (
    <View style={{ flex: 1 }}>
      <CameraScreen
        actions={{ rightButtonText: 'Done', leftButtonText: 'Cancel' }}
        onBottomButtonPressed={(event) => this.onBottomButtonPressed(event)}
        // flashImages={{
        //   // optional, images for flash state
        //   on: require('path/to/image'),
        //   off: require('path/to/image'),
        //   auto: require('path/to/image'),
        // }}
        // cameraFlipImage={require('path/to/image')} // optional, image for flipping camera button
        // captureButtonImage={require('../assets/focus.png')} // optional, image capture button
        // torchOnImage={require('path/to/image')} // optional, image for toggling on flash light
        // torchOffImage={require('path/to/image')} // optional, image for toggling off flash light
        hideControls={false} // (default false) optional, hides camera controls
        showCapturedImageCount={false} // (default false) optional, show count for photos taken during that capture session
      />
    </View>
  );
};

export default CustomCameraScreen;
