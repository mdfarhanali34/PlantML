import React, { useState } from 'react';
import { Button, Image, Text, View } from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';

const HomeScreen = () => {
  const [photo, setPhoto] = useState(null);
  const [response, setResponse] = useState(null);

  const selectImage = async () => {
    try {
      launchImageLibrary(
        {
          mediaType: 'photo',
          includeBase64: false,
        },
        response => {
          if (response.didCancel) {
            console.log('User cancelled image picker');
          } else if (response.errorCode) {
            console.log('ImagePicker Error: ', response.errorMessage);
          } else {
            setPhoto(response.assets[0]);
            uploadImage(response.assets[0]);
          }
        }
      );
    } catch (error) {
      console.log('Error picking image: ', error);
    }
  };

  const uploadImage = async (image) => {
    const formData = new FormData();
    formData.append('image', {
      uri: image.uri,
      type: image.type,
      name: image.fileName,
    });

    try {
      const res = await fetch('https://<server url>/calculate_severity', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const result = await res.json();
      setResponse(result);
      console.log(result);
    } catch (error) {
      console.log('Error uploading image: ', error);
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>Home Screen</Text>
      <Button title="Select Image from Gallery" onPress={selectImage} />
      {photo && (
        <Image source={{ uri: photo.uri }} style={{ width: 200, height: 200, marginTop: 20 }} />
      )}
      {response && (
        <Text style={{ marginTop: 20 }}>{`Severity: ${response.average_percentage}%`}</Text>
      )}
    </View>
  );
};

export default HomeScreen;
