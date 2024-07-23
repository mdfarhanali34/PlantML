import React, { useState } from 'react';
import { Button, Image, Text, View } from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';

const HomeScreen = () => {
  const [photo, setPhoto] = useState(null);

  const selectImage = async () => {
    try {
      launchImageLibrary(
        {
         mediaType: 'photo',
         includeBase64: false,
         maxHeight: 200,
         maxWidth: 200,
        },
         response => {
            console.log(response);
            setPhoto(response);
           },
         );
      // const result = await launchImageLibrary({
      //   mediaType: 'photo',
      //   includeBase64: false,
      //   maxHeight: 200,
      //   maxWidth: 200,
      // });

      // if (result.didCancel) {
      //   console.log('User cancelled image picker');
      // } else if (result.errorCode) {
      //   console.log('ImagePicker Error: ', result.errorMessage);
      // } else {
      //   const source = { uri: result.assets[0].uri };
      //   setSelectedImage(source);
      // }
    } catch (error) {
      console.log('Error picking image: ', error);
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Home Screen</Text>
      <Button title="Select Image from Gallery" onPress={selectImage} />
      {/* {selectedImage && (
        <Image source={selectedImage} style={{ width: 200, height: 200, marginTop: 20 }} />
      )} */}
    </View>
  );
};

export default HomeScreen;
